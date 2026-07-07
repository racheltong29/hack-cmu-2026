import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgar_analyst.ingestion.pipeline import ingest_documents, load_fixture_documents  # noqa: E402
from edgar_analyst.retrieval.embeddings import HashingEmbedder  # noqa: E402
from edgar_analyst.retrieval.hybrid import HybridRetriever  # noqa: E402
from edgar_analyst.retrieval.store import SqliteStore  # noqa: E402


@pytest.fixture(autouse=True)
def _force_stub_llm(monkeypatch):
    # Tests must be hermetic regardless of credentials in the environment.
    monkeypatch.setenv("EDGAR_LLM_PROVIDER", "stub")
    from edgar_analyst.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def documents():
    return load_fixture_documents()


@pytest.fixture()
def indexed():
    """(store, embedder, retriever) with all fixture docs ingested."""
    embedder = HashingEmbedder()
    store = SqliteStore(":memory:")
    ingest_documents(load_fixture_documents(), store, embedder)
    return store, embedder, HybridRetriever(store, embedder)
