"""Ingestion pipeline: documents -> chunks -> embeddings -> store.

Idempotent: chunk IDs are deterministic (`doc_id#n`) and the store upserts,
so re-ingesting a ticker never duplicates vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Document
from ..retrieval.embeddings import Embedder
from .chunker import chunk_document

FIXTURES_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "sample_filings.json"


def load_fixture_documents(path: Path | None = None) -> list[Document]:
    data = json.loads((path or FIXTURES_PATH).read_text())
    return [Document(**d) for d in data["documents"]]


def ingest_documents(docs: list[Document], store, embedder: Embedder) -> int:
    """Chunk, embed, and upsert. Returns the number of chunks written."""
    chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
    if not chunks:
        return 0
    embeddings = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, embeddings)
    return len(chunks)


def ingest_live(ticker: str, store, embedder: Embedder, max_filings: int = 2) -> int:
    """Pull recent real filings from SEC EDGAR and ingest them.

    Requires outbound access to sec.gov; offline environments use fixtures.
    """
    from .edgar_client import EdgarClient

    client = EdgarClient()
    docs = []
    for meta in client.recent_filings(ticker)[:max_filings]:
        text = client.fetch_filing_text(meta["url"])
        docs.append(
            Document(
                doc_id=f"{ticker.upper()}-{meta['form_type']}-{meta['filing_date']}",
                ticker=ticker.upper(),
                form_type=meta["form_type"],
                section="Full filing",
                period=meta["report_date"],
                text=text,
                source_url=meta["url"],
            )
        )
    return ingest_documents(docs, store, embedder)
