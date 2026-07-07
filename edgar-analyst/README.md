# EDGAR Analyst

Multi-agent RAG over SEC filings with hybrid retrieval, a deterministic
citation-verifier guardrail, and an eval harness that gates CI.

Ask a question about a public company; a pipeline of specialist agents
retrieves the relevant filing passages, extracts claims, **verifies every
claim against its cited source**, and synthesizes an answer where each
statement carries a citation back to a specific filing section.

```
router ──► retrieve ──► analyst ──► verifier ──► synthesizer
(route +   (hybrid      (claim      (grounding   (cited answer)
 tickers)   BM25+dense   extraction) guardrail)
            + RRF)
```

## Quickstart (no API key, no services)

```bash
cd edgar-analyst
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python demo.py "Why did NOVA's gross margin decline in fiscal 2025?"
pytest                      # 20 tests, hermetic
python evals/run_evals.py   # eval report + regression gate
```

Everything runs offline by default: bundled fixture filings, a
feature-hashing embedder, SQLite vector storage, and a deterministic
extractive stub LLM. Export `ANTHROPIC_API_KEY` and the same code paths run
on Claude (Haiku for routing, Opus for analysis/synthesis) — the pipeline is
agnostic to which backend it gets.

### API

```bash
uvicorn edgar_analyst.api.main:app --app-dir src
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was NOVA'\''s revenue in fiscal 2025?"}'
curl -N -X POST localhost:8000/ask/stream ...   # SSE: live agent trace + answer
```

### Postgres + pgvector (production storage)

```bash
docker compose up   # pgvector-backed API on :8000
```

Setting `EDGAR_POSTGRES_DSN` switches storage from SQLite brute-force cosine
to HNSW-indexed pgvector; the retrieval code is identical.

### Live SEC EDGAR ingestion

`POST /ingest {"ticker": "AAPL", "live": true}` pulls recent 10-K/10-Q
filings from the free SEC EDGAR APIs (rate-limited, fair-access compliant
User-Agent). Offline environments use the fixture corpus.

## Design decisions

**Hybrid retrieval with RRF.** Filings questions mix rare exact terms
(product names, "Photonix") with paraphrased concepts ("profitability
pressure" → "gross margin decline"). BM25 catches the former, dense
embeddings the latter; Reciprocal Rank Fusion combines them without
calibrating two incompatible score distributions.

**The verifier is code, not a second LLM.** Every figure in a generated
claim must appear verbatim in the cited chunk, and the claim must lexically
overlap its evidence. Grounding guarantees shouldn't depend on a second
model being honest about the first. Unsupported claims are dropped and
counted (`unsupported_claims_removed` in the response); if nothing survives,
the system says so instead of answering.

**Citations quote their support.** The verifier extracts the specific
evidence sentences backing each claim (covering all of its figures), so a
citation can be checked by a human without opening the filing.

**Explicit state machine, swappable frameworks.** The pipeline is a typed
state graph (`agents/graph.py`) with the same node/edge semantics as a
LangGraph `StateGraph`, kept framework-free so every transition is
inspectable. Embedders, vector stores, and LLM backends are protocol-typed
plug points: `HashingEmbedder → bge-m3`, `SQLite → pgvector`,
`stub → Claude` are one-line swaps.

**Deterministic offline mode is a feature, not a fallback.** Tests and evals
run hermetically in CI with stable metrics — no flaky network calls, no
API spend on every push, and eval regressions are attributable to code
changes rather than model drift.

**Cost tiering.** The router runs on Haiku (classification is cheap);
analysis and synthesis run on Opus. Model choices are config, not code.

## Eval harness

`evals/run_evals.py` runs a 15-question golden dataset through the full
pipeline and reports:

| Metric | What it measures | Gate |
|---|---|---|
| `retrieval_recall_at_5` | expected source doc in top-5 retrieved | ≥ 0.80 |
| `retrieval_mrr` | rank of first expected source doc | ≥ 0.60 |
| `citation_precision` | claims whose figures all appear in the cited quote (independent re-check of the verifier) | ≥ 0.90 |
| `answer_keyword_accuracy` | expected figures/terms present in the answer | reported |

The gate exits non-zero below thresholds, and CI
(`.github/workflows/edgar-analyst-ci.yml`) fails the build. Current offline
baseline: recall@5 **1.00**, MRR **0.97**, citation precision **1.00**,
keyword accuracy **0.93**.

## Layout

```
src/edgar_analyst/
  ingestion/    EDGAR client (submissions API, HTML→text), sentence-aware chunker, idempotent pipeline
  retrieval/    embedders (hashing / sentence-transformers), stores (SQLite / pgvector), hybrid+RRF
  agents/       typed state-machine pipeline with per-node trace events
  llm/          Anthropic client (streaming, structured output) + deterministic stub
  api/          FastAPI: /ask, /ask/stream (SSE), /ingest, /health
evals/          golden dataset + gated metrics runner
tests/          20 hermetic unit/integration tests
fixtures/       synthetic sample filings (offline corpus)
```

## Roadmap

- ColPali/ColQwen2 visual document retrieval for table-heavy filing pages
- Earnings-call audio: Whisper ASR + diarization as a second evidence source
- XBRL ground-truth checks for numeric claims (EDGAR companyfacts API)
- Langfuse tracing with per-query cost dashboards
- Cross-encoder reranking stage between fusion and the analyst

The full project plan, JD mapping, and 12-week milestones are in
[`../AI_ENGINEERING_PORTFOLIO_PROJECT.md`](../AI_ENGINEERING_PORTFOLIO_PROJECT.md).
