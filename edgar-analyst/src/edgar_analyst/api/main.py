"""FastAPI service.

POST /ask            — full pipeline, JSON answer with citations
POST /ask/stream     — same pipeline, SSE trace of agent events + final answer
POST /ingest         — (re)ingest fixture documents, or live EDGAR with ?live=1
GET  /health         — index size + backend info
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents.graph import AnalystPipeline
from ..config import get_settings
from ..ingestion.pipeline import ingest_documents, ingest_live, load_fixture_documents
from ..llm.client import get_llm
from ..models import Answer
from ..retrieval.embeddings import get_embedder
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.store import get_store


class AskRequest(BaseModel):
    question: str
    ticker: str | None = None


class IngestRequest(BaseModel):
    ticker: str | None = None
    live: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = get_embedder()
    store = get_store(embedder.dim)
    retriever = HybridRetriever(store, embedder)
    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.llm = get_llm()
    if store.count() == 0:
        ingest_documents(load_fixture_documents(), store, embedder)
        retriever.invalidate()
    yield


app = FastAPI(title="EDGAR Analyst", version="0.1.0", lifespan=lifespan)


def _pipeline(app_: FastAPI) -> AnalystPipeline:
    known = {c.ticker for c in app_.state.store.all_chunks()}
    return AnalystPipeline(app_.state.retriever, app_.state.llm, known)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "chunks_indexed": app.state.store.count(),
        "llm": app.state.llm.name,
        "store": type(app.state.store).__name__,
        "embedder": type(app.state.embedder).__name__,
        "models": {"router": s.router_model, "synthesis": s.synthesis_model},
    }


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    return _pipeline(app).run(req.question, req.ticker)


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    pipeline = _pipeline(app)

    def event_source():
        for item in pipeline.stream(req.question, req.ticker):
            if isinstance(item, Answer):
                yield f"event: answer\ndata: {item.model_dump_json()}\n\n"
            else:
                yield f"event: trace\ndata: {item.model_dump_json()}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if req.live:
        if not req.ticker:
            raise HTTPException(status_code=422, detail="ticker is required for live ingestion")
        try:
            n = ingest_live(req.ticker, app.state.store, app.state.embedder)
        except Exception as exc:  # network/policy failures surface as 502, not 500
            raise HTTPException(status_code=502, detail=f"EDGAR fetch failed: {exc}") from exc
    else:
        docs = load_fixture_documents()
        if req.ticker:
            docs = [d for d in docs if d.ticker == req.ticker.upper()]
        n = ingest_documents(docs, app.state.store, app.state.embedder)
    app.state.retriever.invalidate(req.ticker)
    return {"chunks_ingested": n, "total_chunks": app.state.store.count()}
