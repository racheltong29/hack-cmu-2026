from edgar_analyst.agents.graph import AnalystPipeline
from edgar_analyst.llm.client import StubLLM
from edgar_analyst.models import AgentEvent, Answer


def make_pipeline(indexed) -> AnalystPipeline:
    store, _, retriever = indexed
    tickers = {c.ticker for c in store.all_chunks()}
    return AnalystPipeline(retriever, StubLLM(), tickers)


def test_end_to_end_answer_with_citations(indexed):
    pipeline = make_pipeline(indexed)
    answer = pipeline.run("What was NOVA's total revenue in fiscal 2025?")
    assert "20.1" in answer.answer
    assert answer.citations
    assert all(c.ticker == "NOVA" for c in answer.citations)


def test_router_extracts_ticker_and_risk_route(indexed):
    pipeline = make_pipeline(indexed)
    answer = pipeline.run("What are the biggest risk factors for BLDR?")
    assert answer.route == "risk"
    assert all(h.chunk.ticker == "BLDR" for h in answer.hits)


def test_stream_emits_full_trace(indexed):
    pipeline = make_pipeline(indexed)
    items = list(pipeline.stream("What was BLDR's net revenue in fiscal 2025?"))
    events = [i for i in items if isinstance(i, AgentEvent)]
    answers = [i for i in items if isinstance(i, Answer)]
    nodes = [e.node for e in events if e.status == "finished"]
    assert nodes == ["router", "retrieve", "analyst", "verifier", "synthesizer"]
    assert len(answers) == 1


def test_verifier_rejects_fabricated_numbers(indexed):
    pipeline = make_pipeline(indexed)
    supported = pipeline._claim_supported(
        "Revenue was $99.9 billion in fiscal 2025.",
        "Fiscal 2025 revenue was $20.1 billion, an increase of 42 percent.",
    )
    assert supported is False


def test_verifier_accepts_grounded_claim(indexed):
    pipeline = make_pipeline(indexed)
    supported = pipeline._claim_supported(
        "Fiscal 2025 revenue was $20.1 billion, an increase of 42 percent.",
        "Fiscal 2025 revenue was $20.1 billion, an increase of 42 percent from $14.2 billion "
        "in fiscal 2024, driven primarily by Data Center demand.",
    )
    assert supported is True


def test_unanswerable_question_refuses_gracefully(indexed):
    pipeline = make_pipeline(indexed)
    answer = pipeline.run("What is the airspeed velocity of an unladen swallow?")
    assert "No sufficiently supported evidence" in answer.answer
    assert answer.citations == []
