from edgar_analyst.retrieval.embeddings import HashingEmbedder


def test_ingestion_is_idempotent(indexed):
    store, embedder, retriever = indexed
    from edgar_analyst.ingestion.pipeline import ingest_documents, load_fixture_documents

    before = store.count()
    ingest_documents(load_fixture_documents(), store, embedder)
    assert store.count() == before


def test_hybrid_finds_revenue_chunk(indexed):
    _, _, retriever = indexed
    hits = retriever.search("What was total revenue in fiscal 2025?", ticker="NOVA", top_k=5)
    assert hits
    assert any(h.chunk.doc_id == "NOVA-10K-FY2025-item7" for h in hits)


def test_ticker_filter_respected(indexed):
    _, _, retriever = indexed
    hits = retriever.search("revenue growth", ticker="BLDR", top_k=8)
    assert hits
    assert all(h.chunk.ticker == "BLDR" for h in hits)


def test_lexical_leg_catches_rare_exact_term(indexed):
    # "Photonix" appears once; BM25 must surface it even if dense embedding is weak.
    _, _, retriever = indexed
    hits = retriever.search("Photonix acquisition", top_k=5)
    assert any("Photonix" in h.chunk.text for h in hits)


def test_embedder_similarity_ordering():
    emb = HashingEmbedder()
    vecs = emb.embed(
        [
            "gross margin declined due to inventory charges",
            "gross margin fell because of inventory write-downs and charges",
            "the quick brown fox jumps over the lazy dog",
        ]
    )
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated
