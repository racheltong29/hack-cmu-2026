"""Eval harness with CI regression gate.

Metrics:
  retrieval_recall@5   fraction of questions where an expected source document
                       appears in the top-5 retrieved chunks
  retrieval_mrr        reciprocal rank of the first expected document
  citation_precision   fraction of answer claims whose figures all appear
                       verbatim in the cited chunk (independent re-check of
                       the verifier — measures end-to-end grounding)
  answer_keyword_acc   fraction of questions whose answer contains the
                       expected key figures/terms

Exit code is non-zero when any gated metric falls below its threshold
(EDGAR_EVAL_MIN_*), which is what fails the CI job.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgar_analyst.agents.graph import AnalystPipeline  # noqa: E402
from edgar_analyst.config import get_settings  # noqa: E402
from edgar_analyst.ingestion.pipeline import ingest_documents, load_fixture_documents  # noqa: E402
from edgar_analyst.llm.client import get_llm  # noqa: E402
from edgar_analyst.retrieval.embeddings import get_embedder  # noqa: E402
from edgar_analyst.retrieval.hybrid import HybridRetriever  # noqa: E402
from edgar_analyst.retrieval.store import SqliteStore  # noqa: E402

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def build_pipeline() -> AnalystPipeline:
    embedder = get_embedder()
    store = SqliteStore(":memory:")
    docs = load_fixture_documents()
    ingest_documents(docs, store, embedder)
    retriever = HybridRetriever(store, embedder)
    return AnalystPipeline(retriever, get_llm(), {d.ticker for d in docs})


def load_golden() -> list[dict]:
    path = Path(__file__).parent / "golden.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate() -> dict:
    pipeline = build_pipeline()
    cases = load_golden()

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    keyword_hits = 0
    claims_checked = 0
    claims_grounded = 0
    failures: list[str] = []

    for case in cases:
        answer = pipeline.run(case["question"])
        retrieved_docs = [h.chunk.doc_id for h in answer.hits]
        expected = set(case["expected_doc_ids"])

        # recall@5 / MRR over source documents
        top5 = retrieved_docs[:5]
        if expected & set(top5):
            recall_hits += 1
        else:
            failures.append(f"recall@5 miss: {case['id']} (got {top5})")
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # answer keyword accuracy
        if all(kw.lower() in answer.answer.lower() for kw in case["expected_keywords"]):
            keyword_hits += 1
        else:
            failures.append(f"keyword miss: {case['id']} (answer: {answer.answer[:120]!r})")

        # citation precision: independently re-check every claim's figures
        # against the quoted evidence for its citation
        for claim in answer.claims:
            claims_checked += 1
            evidence_numbers = set(_NUMBER_RE.findall(claim.citation.quote))
            claim_numbers = set(_NUMBER_RE.findall(claim.claim))
            if claim_numbers <= evidence_numbers:
                claims_grounded += 1
            else:
                failures.append(f"ungrounded claim in {case['id']}: {claim.claim[:100]!r}")

    n = len(cases)
    return {
        "cases": n,
        "retrieval_recall_at_5": recall_hits / n,
        "retrieval_mrr": sum(reciprocal_ranks) / n,
        "answer_keyword_accuracy": keyword_hits / n,
        "citation_precision": (claims_grounded / claims_checked) if claims_checked else 0.0,
        "claims_checked": claims_checked,
        "failures": failures,
    }


def main() -> int:
    s = get_settings()
    report = evaluate()

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report["failures"]:
        print("\nFailures:", file=sys.stderr)
        for f in report["failures"]:
            print(f"  - {f}", file=sys.stderr)

    gates = [
        ("retrieval_recall_at_5", s.eval_min_recall_at_5),
        ("retrieval_mrr", s.eval_min_mrr),
        ("citation_precision", s.eval_min_citation_precision),
    ]
    failed = [(name, report[name], threshold) for name, threshold in gates if report[name] < threshold]
    if failed:
        print("\nEVAL GATE FAILED:", file=sys.stderr)
        for name, value, threshold in failed:
            print(f"  {name} = {value:.3f} < required {threshold:.3f}", file=sys.stderr)
        return 1
    print("\nEval gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
