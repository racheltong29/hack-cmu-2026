"""Hybrid retrieval: BM25 (lexical) + dense (semantic), fused with
Reciprocal Rank Fusion. RRF is rank-based, so the two score distributions
never need calibrating against each other.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..models import Chunk, RetrievalHit
from .embeddings import Embedder, tokenize


class HybridRetriever:
    def __init__(self, store, embedder: Embedder):
        self._store = store
        self._embedder = embedder
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}

    def invalidate(self, ticker: str | None = None) -> None:
        """Drop cached BM25 indexes after ingestion."""
        if ticker:
            self._bm25_cache.pop(ticker.upper(), None)
        self._bm25_cache.pop("*", None)

    def _bm25_index(self, ticker: str | None) -> tuple[BM25Okapi, list[Chunk]]:
        key = (ticker or "*").upper()
        if key not in self._bm25_cache:
            chunks = self._store.all_chunks(ticker)
            corpus = [tokenize(c.text) for c in chunks] or [["_empty_"]]
            self._bm25_cache[key] = (BM25Okapi(corpus), chunks)
        return self._bm25_cache[key]

    def search(self, query: str, ticker: str | None = None, top_k: int | None = None) -> list[RetrievalHit]:
        s = get_settings()
        top_k = top_k or s.retrieval_top_k
        pool = top_k * 3  # over-fetch each leg, fuse, then cut

        # Lexical leg
        bm25, chunks = self._bm25_index(ticker)
        lexical_ranked: list[str] = []
        if chunks:
            scores = bm25.get_scores(tokenize(query))
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:pool]
            lexical_ranked = [chunks[i].chunk_id for i in order if scores[i] > 0]

        # Dense leg
        qvec = self._embedder.embed([query])[0]
        dense = self._store.search(qvec, top_k=pool, ticker=ticker)
        dense_ranked = [c.chunk_id for c, _ in dense]

        # Reciprocal Rank Fusion
        fused: dict[str, float] = {}
        for ranked in (lexical_ranked, dense_ranked):
            for rank, chunk_id in enumerate(ranked):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (s.rrf_k + rank + 1)

        by_id = {c.chunk_id: c for c in chunks}
        by_id.update({c.chunk_id: c for c, _ in dense})
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievalHit(chunk=by_id[cid], score=score, source="fused")
            for cid, score in top
            if cid in by_id
        ]
