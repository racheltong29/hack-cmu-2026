"""Multi-agent pipeline as an explicit typed state machine.

    router ──► retrieve ──► analyst ──► verifier ──► synthesizer

Each node is a pure function over `PipelineState` and emits trace events,
which the API streams to clients as SSE. The design mirrors a LangGraph
`StateGraph` (typed state, node functions, conditional edges) without the
framework dependency, so the control flow is fully inspectable — swapping
in LangGraph is a mechanical change confined to this module.

The verifier is deterministic code, not an LLM: every figure in a claim
must literally appear in the cited chunk, and the claim must lexically
overlap its evidence. Grounding guarantees shouldn't depend on a second
model being honest about the first.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..config import get_settings
from ..llm.client import LLMClient
from ..models import AgentEvent, Answer, Citation, RetrievalHit, VerifiedClaim
from ..retrieval.embeddings import tokenize
from ..retrieval.hybrid import HybridRetriever

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["claim", "chunk_id"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

ROUTER_SYSTEM = (
    "You are a query router for a financial-filings QA system. "
    "Respond with exactly one word: filings, risk, or comparison."
)
ANALYST_SYSTEM = (
    "You are a financial filings analyst. Extract factual claims that answer the question, "
    "each grounded in exactly one evidence block. Copy figures exactly; never infer numbers "
    "that are not present in the evidence."
)
SYNTH_SYSTEM = (
    "You are a financial research writer. Compose a concise answer using ONLY the verified "
    "claims provided, keeping each claim's [n] citation marker attached to its statement. "
    "Do not introduce any fact that is not in the claims."
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class PipelineState:
    question: str
    tickers: list[str] = field(default_factory=list)
    route: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    raw_claims: list[dict] = field(default_factory=list)
    verified: list[VerifiedClaim] = field(default_factory=list)
    answer_text: str = ""
    model_used: str = ""


class AnalystPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, known_tickers: set[str]):
        self._retriever = retriever
        self._llm = llm
        self._known_tickers = known_tickers
        self._settings = get_settings()

    # ---------------------------------------------------------------- run
    def run(self, question: str, ticker: str | None = None) -> Answer:
        """Run to completion, discarding the trace."""
        result: Answer | None = None
        for event in self.stream(question, ticker):
            if isinstance(event, Answer):
                result = event
        assert result is not None
        return result

    def stream(self, question: str, ticker: str | None = None) -> Iterator[AgentEvent | Answer]:
        """Run the graph, yielding trace events and finally the Answer."""
        state = PipelineState(question=question)
        if ticker:
            state.tickers = [ticker.upper()]

        for node_name, node_fn in (
            ("router", self._node_router),
            ("retrieve", self._node_retrieve),
            ("analyst", self._node_analyst),
            ("verifier", self._node_verifier),
            ("synthesizer", self._node_synthesizer),
        ):
            yield AgentEvent(node=node_name, status="started")
            detail = node_fn(state)
            yield AgentEvent(node=node_name, status="finished", detail=detail)

        yield self._build_answer(state)

    # -------------------------------------------------------------- nodes
    def _node_router(self, state: PipelineState) -> dict:
        # Ticker extraction is deterministic string matching against the index —
        # no reason to spend model tokens on it.
        if not state.tickers:
            words = set(re.findall(r"[A-Za-z]+", state.question.upper()))
            state.tickers = sorted(words & self._known_tickers)
        route = self._llm.complete(
            system=ROUTER_SYSTEM,
            prompt=f"TASK: route\nQUESTION: {state.question}",
            model=self._settings.router_model,
            max_tokens=16,
        ).strip().lower()
        state.route = route if route in ("filings", "risk", "comparison") else "filings"
        return {"route": state.route, "tickers": state.tickers}

    def _node_retrieve(self, state: PipelineState) -> dict:
        tickers = state.tickers or [None]  # type: ignore[list-item]
        per_ticker = max(2, self._settings.retrieval_top_k // len(tickers))
        seen: set[str] = set()
        for t in tickers:
            query = state.question
            if state.route == "risk":
                query += " risk factors headwinds"
            for hit in self._retriever.search(query, ticker=t, top_k=per_ticker):
                if hit.chunk.chunk_id not in seen:
                    seen.add(hit.chunk.chunk_id)
                    state.hits.append(hit)
        return {"hits": len(state.hits), "chunk_ids": [h.chunk.chunk_id for h in state.hits]}

    def _node_analyst(self, state: PipelineState) -> dict:
        evidence = "\n".join(f"[{h.chunk.chunk_id}]\n{h.chunk.text}" for h in state.hits)
        result = self._llm.complete_json(
            system=ANALYST_SYSTEM,
            prompt=(
                f"TASK: extract_claims\nQUESTION: {state.question}\n\nEVIDENCE BLOCKS:\n{evidence}"
            ),
            schema=CLAIMS_SCHEMA,
            model=self._settings.synthesis_model,
        )
        state.raw_claims = result.get("claims", [])
        return {"claims": len(state.raw_claims)}

    def _node_verifier(self, state: PipelineState) -> dict:
        by_id = {h.chunk.chunk_id: h.chunk for h in state.hits}
        seen_claims: set[str] = set()
        for raw in state.raw_claims:
            chunk = by_id.get(raw.get("chunk_id", ""))
            claim_text = raw.get("claim", "").strip()
            if not chunk or not claim_text:
                continue
            # Overlapping chunks can yield the same sentence twice — keep one.
            dedupe_key = " ".join(tokenize(claim_text))
            if dedupe_key in seen_claims:
                continue
            seen_claims.add(dedupe_key)
            supported = self._claim_supported(claim_text, chunk.text)
            quote = self._supporting_quote(claim_text, chunk.text)
            state.verified.append(
                VerifiedClaim(
                    claim=claim_text,
                    citation=Citation(
                        chunk_id=chunk.chunk_id,
                        ticker=chunk.ticker,
                        form_type=chunk.form_type,
                        section=chunk.section,
                        period=chunk.period,
                        quote=quote,
                    ),
                    supported=supported,
                    evidence=quote if supported else "",
                )
            )
        kept = sum(1 for v in state.verified if v.supported)
        return {"claims_in": len(state.verified), "claims_supported": kept}

    @staticmethod
    def _supporting_quote(claim: str, evidence: str, max_sentences: int = 3) -> str:
        """Select the evidence sentences that actually back the claim.

        Sentences are ranked by lexical overlap with the claim and accumulated
        until every figure in the claim is covered, so the rendered citation
        quote is verifiable on its own.
        """
        sentences = re.split(r"(?<=[.!?])\s+", evidence.strip())
        claim_toks = set(tokenize(claim))
        claim_numbers = set(_NUMBER_RE.findall(claim))
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: -len(claim_toks & set(tokenize(sentences[i]))),
        )
        picked: list[int] = []
        covered: set[str] = set()
        for i in ranked[: max_sentences * 2]:
            picked.append(i)
            covered |= set(_NUMBER_RE.findall(sentences[i]))
            if len(picked) >= max_sentences or claim_numbers <= covered:
                if claim_numbers <= covered:
                    break
        picked.sort()
        quote = " ".join(sentences[i] for i in picked[:max_sentences])
        return quote or evidence[:240]

    @staticmethod
    def _claim_supported(claim: str, evidence: str) -> bool:
        # Rule 1: every figure in the claim must appear verbatim in the evidence.
        evidence_numbers = set(_NUMBER_RE.findall(evidence))
        for num in _NUMBER_RE.findall(claim):
            if num not in evidence_numbers:
                return False
        # Rule 2: the claim's content words must substantially overlap the evidence.
        claim_toks = set(tokenize(claim))
        if not claim_toks:
            return False
        overlap = len(claim_toks & set(tokenize(evidence))) / len(claim_toks)
        return overlap >= 0.6

    def _node_synthesizer(self, state: PipelineState) -> dict:
        supported = [v for v in state.verified if v.supported]
        if not supported:
            state.answer_text = (
                "No sufficiently supported evidence was found in the indexed filings "
                "to answer this question."
            )
            state.model_used = self._llm.name
            return {"answer_chars": len(state.answer_text)}
        claims_block = "\n".join(f"CLAIM [{i + 1}]: {v.claim}" for i, v in enumerate(supported))
        state.answer_text = self._llm.complete(
            system=SYNTH_SYSTEM,
            prompt=f"TASK: synthesize\nQUESTION: {state.question}\n\n{claims_block}",
            model=self._settings.synthesis_model,
            max_tokens=1024,
        ).strip()
        state.model_used = f"{self._llm.name}:{self._settings.synthesis_model}"
        return {"answer_chars": len(state.answer_text)}

    # ------------------------------------------------------------- output
    @staticmethod
    def _build_answer(state: PipelineState) -> Answer:
        supported = [v for v in state.verified if v.supported]
        return Answer(
            question=state.question,
            route=state.route,
            answer=state.answer_text,
            claims=supported,
            citations=[v.citation for v in supported],
            hits=state.hits,
            model_used=state.model_used,
            unsupported_claims_removed=len(state.verified) - len(supported),
        )
