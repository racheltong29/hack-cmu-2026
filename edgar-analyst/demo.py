"""Terminal demo: ask a question, watch the agent trace, get a cited answer.

    python demo.py "What was NOVA's revenue in fiscal 2025?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from edgar_analyst.agents.graph import AnalystPipeline
from edgar_analyst.ingestion.pipeline import ingest_documents, load_fixture_documents
from edgar_analyst.llm.client import get_llm
from edgar_analyst.models import Answer
from edgar_analyst.retrieval.embeddings import get_embedder
from edgar_analyst.retrieval.hybrid import HybridRetriever
from edgar_analyst.retrieval.store import SqliteStore


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What was NOVA's total revenue in fiscal 2025?"

    embedder = get_embedder()
    store = SqliteStore(":memory:")
    docs = load_fixture_documents()
    n = ingest_documents(docs, store, embedder)
    llm = get_llm()
    print(f"[index] {n} chunks from {len(docs)} filing sections | llm={llm.name}\n")

    pipeline = AnalystPipeline(HybridRetriever(store, embedder), llm, {d.ticker for d in docs})
    print(f"Q: {question}\n")
    for item in pipeline.stream(question):
        if isinstance(item, Answer):
            print(f"\nA: {item.answer}\n")
            for i, c in enumerate(item.citations, 1):
                print(f"  [{i}] {c.ticker} {c.form_type} {c.period} — {c.section}")
                print(f"      “{c.quote[:140]}…”")
            if item.unsupported_claims_removed:
                print(f"\n  (verifier removed {item.unsupported_claims_removed} unsupported claim(s))")
        elif item.status == "finished":
            print(f"  ▸ {item.node:<12} {item.detail}")


if __name__ == "__main__":
    main()
