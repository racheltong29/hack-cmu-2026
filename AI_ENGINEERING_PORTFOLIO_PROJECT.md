# EDGAR Analyst — Multimodal Financial-Filing Intelligence Platform

A portfolio project plan for a mid-level SWE (2–6 YOE) pivoting into AI engineering.
Target: shippable solo in 1–3 months, production-grade patterns throughout, built
on real public data (SEC EDGAR + earnings-call audio), not a toy dataset.

---

## 1. What it is and what problem it solves

**EDGAR Analyst** is a multi-agent research copilot that answers hard questions
about public companies by reading what humans actually read: SEC filings
(10-K/10-Q/8-K — dense PDFs full of tables, charts, and footnotes) and earnings
calls (hour-long audio). Ask it *"How did NVDA's data-center margin trend over
the last 4 quarters, and what risks did management flag on the calls?"* and it:

1. Retrieves the right filing sections and call segments (hybrid + visual retrieval),
2. Fans out to specialist agents (filings analyst, transcript analyst, quant/table agent),
3. Synthesizes a cited answer where **every claim links to a page/timestamp**,
4. Logs the whole run with traces, cost, and eval scores.

**Real-world problem:** analysts and retail investors drown in disclosure
documents. A 10-K averages 100+ pages; earnings calls add another hour per
quarter per company. Existing tools either do naive text-chunk RAG (which
destroys tables and charts — where the actual numbers live) or are closed,
expensive terminals. This project demonstrates the hard version: multimodal
retrieval over documents-as-images, audio pipelines, and verifiable citations.

**Why it stands out in 2026 hiring:** it hits the exact cluster interviewers
probe for — RAG that goes beyond `text-splitter + cosine similarity`, agent
orchestration with real routing decisions, an eval harness with regression
gates in CI, and cost/latency observability. It's also demo-friendly: anyone
can type a ticker and watch agents work.

---

## 2. HuggingFace tasks it integrates

| HF Task | Where it's used | Example models |
|---|---|---|
| **Visual Document Retrieval** | Retrieve filing pages as *images* so tables/charts survive (ColPali-style late interaction) | `vidore/colqwen2-v1.0`, `vidore/colpali-v1.3` |
| **Document Question Answering** | Extract answers from retrieved page images | `Qwen/Qwen2.5-VL-7B-Instruct` |
| **Image-Text-to-Text (VLM)** | Chart/figure understanding inside filings; describe exhibits | Qwen2.5-VL, `HuggingFaceTB/SmolVLM` |
| **Automatic Speech Recognition** | Transcribe earnings-call audio with timestamps + diarization | `openai/whisper-large-v3-turbo` + pyannote |
| **Sentence Similarity / Feature Extraction** | Dense leg of hybrid text retrieval | `BAAI/bge-m3`, `nomic-ai/nomic-embed-text-v1.5` |
| **Text Ranking** | Cross-encoder reranking of fused candidates | `BAAI/bge-reranker-v2-m3` |
| **Table Question Answering** | Numeric QA over extracted financial tables | table extraction → SQL/pandas agent (TAPAS as baseline) |
| **Text Classification** | Query router (which agent/index?) + finance sentiment on call segments | fine-tuned `distilbert`, `ProsusAI/finbert` |
| **Summarization** | Map-reduce section and call summaries feeding the synthesis agent | LLM-based, with faithfulness evals |
| **Text Generation** | The agents themselves | Claude / any hosted LLM + one self-hosted OSS model |

Using **Visual Document Retrieval** as the retrieval backbone is the
differentiator — most portfolio RAG projects have never touched it, and it's
the current state of the art for exactly this document type.

---

## 3. Recommended tech stack

- **Orchestration:** **LangGraph** (supervisor + specialist agents as a typed
  graph with checkpointing/HITL interrupts). LangGraph over vanilla LangChain
  chains — interviewers want to see explicit state machines, not chains.
- **Ingestion / parsing:** `unstructured` or `docling` for PDF→structure,
  `pdf2image` for page images, **SEC EDGAR full-text + submissions APIs**
  (free, no key) via `edgartools`; earnings audio from company IR pages or the
  free tier of an earnings-call API (e.g. API Ninjas transcripts as fallback).
- **Retrieval:**
  - **pgvector on Postgres** as the primary store (one database for vectors,
    metadata, and eval results — a deliberately boring, production-realistic choice).
  - Hybrid search: BM25 (Postgres `tsvector` or OpenSearch) + dense (bge-m3)
    fused with Reciprocal Rank Fusion → cross-encoder rerank.
  - ColQwen2 multivector page-image index (pgvector halfvec or Qdrant
    multivector) for the visual-retrieval leg.
- **Model serving:** hosted LLM (Claude API) for agents; **one self-hosted
  model** (e.g. Whisper + reranker on a small GPU via HF `text-embeddings-inference`
  / `vLLM`) to show you can run inference infrastructure, not just call APIs.
- **Evals:** **promptfoo** or **Ragas + pytest** for retrieval metrics
  (recall@k, MRR) and generation metrics (faithfulness, citation precision),
  LLM-as-judge with a rubric, golden dataset of ~150 Q/A pairs, **run in CI on
  every PR with regression thresholds**.
- **LLMOps:** **Langfuse** (self-hostable) for tracing, per-query cost/latency,
  prompt versioning, and online feedback capture. OpenTelemetry export.
- **API & app:** FastAPI (async, SSE streaming) + a thin Next.js or Streamlit
  UI showing the agent graph live and click-through citations.
- **Infra:** Docker Compose for local, one-command deploy to Fly.io/Railway/EC2;
  GitHub Actions for lint + unit tests + eval gate; Terraform optional stretch.

*(LlamaIndex and Pinecone are fine substitutes — LlamaIndex if you prefer its
ingestion abstractions, Pinecone/Qdrant if you'd rather show a managed vector
DB — but LangGraph + pgvector is the strongest interview story: explicit
control flow and boring, scalable storage.)*

---

## 4. System design (what makes it beyond-a-demo)

```
                        ┌────────────────────────────────────────┐
 EDGAR API / IR audio ─►│ Ingestion workers (queue: Redis/Celery)│
                        │  PDF→pages→images  audio→ASR→diarized  │
                        │  tables→Postgres    chunks→embeddings  │
                        └───────────────┬────────────────────────┘
                                        ▼
                 Postgres + pgvector (text, multivector pages, tables, metadata)
                                        ▲
        ┌───────────────────────────────┴───────────────────────────┐
        │                    LangGraph supervisor                   │
        │  router (classifier) ─► filings agent / calls agent /     │
        │  quant-table agent ─► citation verifier ─► synthesizer    │
        └───────────────┬───────────────────────────────────────────┘
                        ▼
      FastAPI (SSE) ──► UI (live agent trace, cited answer)
                        │
                        └──► Langfuse traces ──► eval harness ──► CI gate
```

Design decisions worth writing up in the README (these are the interview
talking points):

- **Documents as images, not just text.** Late-interaction visual retrieval
  vs. OCR-and-chunk, with a benchmark table showing recall@5 on table-heavy
  questions for each approach.
- **A citation-verifier node** that re-checks every generated claim against
  its cited page/timestamp before the answer ships — grounded-generation
  guardrail, measurably reduces hallucination rate.
- **Async ingestion with idempotent workers** — re-running a ticker doesn't
  duplicate vectors; new filings are picked up by a scheduled job.
- **Cost tiering:** router sends easy lookups to a cheap/small model, hard
  synthesis to the frontier model; dashboard shows $/query by route.
- **Scalability story:** stateless API pods, queue-backed ingestion, pgvector
  HNSW indexes, embedding cache; load-test numbers (Locust) in the README.

### Eval harness (the section most portfolios are missing)

- Golden set: ~150 questions across 8–10 tickers, labeled with source
  page/timestamp (build ~50 by hand, LLM-assist the rest, hand-verify).
- **Retrieval evals:** recall@k, MRR — text-only vs. hybrid vs. hybrid+visual.
- **Generation evals:** faithfulness & answer-relevance (Ragas), citation
  precision (does the cited page actually contain the claim?), numeric
  accuracy on table questions (exact-match against XBRL ground truth — EDGAR
  gives you structured numbers to check against for free).
- **CI regression gate:** PR fails if faithfulness or recall drops >2 points.
- **A/B in traces:** prompt versions tagged in Langfuse; compare online.

---

## 5. How it maps to real AI-engineering JDs

| Common JD line (2025–26 postings) | Where this project proves it |
|---|---|
| "Design and productionize RAG pipelines" | Hybrid + multimodal retrieval, reranking, ingestion workers, freshness jobs |
| "Build agentic workflows / multi-agent systems" | LangGraph supervisor with router, specialists, verifier, HITL checkpoint |
| "Evaluate LLM systems; own quality metrics" | Golden dataset, Ragas/promptfoo, LLM-judge rubric, CI eval gate |
| "LLMOps: observability, cost, latency" | Langfuse tracing, $/query dashboards, model-tiering router, load tests |
| "Fine-tune / deploy open-source models" | Self-hosted Whisper + reranker; optional LoRA on the router classifier |
| "Work with unstructured/multimodal data" | PDFs-as-images, tables, charts, hour-long audio with diarization |
| "Ship full-stack ML products" | FastAPI + streaming UI + Docker + CI/CD, one-command deploy |

That's essentially the entire skills matrix of a mid-level "AI Engineer" or
"GenAI Engineer" posting, demonstrated in one coherent artifact.

---

## 6. Resume-ready impact metrics & portfolio value

Instrument the system so these bullets are *measured, not invented* (targets
below are realistic; report your actuals):

- "Built a multimodal RAG platform over **10K+ SEC filing pages and 100+ hours
  of earnings audio**; visual document retrieval (ColQwen2) improved recall@5
  on table-heavy queries **from ~0.55 to ~0.85** vs. text-only chunking."
- "Designed a 5-node LangGraph multi-agent system with a citation-verifier
  guardrail, cutting unsupported claims **from ~18% to <4%** on a 150-question
  golden set."
- "Stood up an LLM eval harness (Ragas + promptfoo) gating CI; caught **100%
  of seeded prompt regressions** before merge."
- "Reduced cost per query **~60%** via model-tier routing and embedding caching
  while holding P95 latency under **8s** for full multi-agent runs."
- "Self-hosted Whisper-large-v3-turbo + a cross-encoder reranker on a single
  GPU, sustaining **N req/s** at **X ms** P95 (load-tested with Locust)."

**Portfolio packaging (do not skip):** public repo with an architecture
diagram, a 3-minute demo video, a hosted live demo (2–3 pre-indexed tickers to
cap cost), the eval report as a rendered page, and 2–3 blog posts ("Why I
retrieve pages as images", "Putting LLM evals in CI", "What a $/query
dashboard taught me"). The write-ups are what make recruiters and hiring
managers actually *see* the engineering.

---

## 7. 12-week solo roadmap

| Weeks | Milestone |
|---|---|
| 1–2 | EDGAR ingestion → Postgres/pgvector; baseline text RAG + FastAPI; Docker Compose |
| 3–4 | Hybrid search + reranker; golden dataset v1; retrieval evals running locally |
| 5–6 | ColQwen2 page-image index + VLM answerer; benchmark vs. text baseline |
| 7–8 | Audio pipeline (ASR + diarization + sentiment); LangGraph supervisor + specialist agents |
| 9 | Citation verifier, table/quant agent with XBRL ground-truth checks |
| 10 | Langfuse tracing, cost dashboards, model-tier routing; eval gate in CI |
| 11 | UI polish (live agent trace, click-to-source citations), load testing, deploy |
| 12 | Eval report, demo video, blog posts, README deep-dive |

**Scope valves** if time runs short (in cut order): diarization → plain ASR;
self-hosted models → hosted APIs with a note; 8 tickers → 3; Next.js →
Streamlit. The eval harness and citation verifier are *never* cut — they're
the differentiators.
