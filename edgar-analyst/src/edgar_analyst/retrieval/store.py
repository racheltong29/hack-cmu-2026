"""Chunk + vector storage.

`SqliteStore` is the default: one file, zero services, vectors stored as
float32 blobs with brute-force cosine search (fine up to ~100K chunks).
`PgVectorStore` (optional extra) is the production path: HNSW-indexed
pgvector on Postgres, same interface.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..models import Chunk


class SqliteStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL,
                embedding BLOB NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON chunks(ticker)")
        self._conn.commit()

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        rows = [
            (c.chunk_id, c.ticker, c.model_dump_json(), embeddings[i].astype(np.float32).tobytes())
            for i, c in enumerate(chunks)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, ticker, payload, embedding) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def all_chunks(self, ticker: str | None = None) -> list[Chunk]:
        if ticker:
            cur = self._conn.execute("SELECT payload FROM chunks WHERE ticker = ?", (ticker.upper(),))
        else:
            cur = self._conn.execute("SELECT payload FROM chunks")
        return [Chunk(**json.loads(row[0])) for row in cur.fetchall()]

    def search(self, query_vec: np.ndarray, top_k: int = 8, ticker: str | None = None) -> list[tuple[Chunk, float]]:
        if ticker:
            cur = self._conn.execute(
                "SELECT payload, embedding FROM chunks WHERE ticker = ?", (ticker.upper(),)
            )
        else:
            cur = self._conn.execute("SELECT payload, embedding FROM chunks")
        rows = cur.fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        scores = matrix @ query_vec.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        return [(Chunk(**json.loads(rows[i][0])), float(scores[i])) for i in order]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


class PgVectorStore:
    """Postgres + pgvector backend (requires the `pg` extra and a running DB)."""

    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                payload JSONB NOT NULL,
                embedding vector({dim}) NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON chunks(ticker)")

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        with self._conn.cursor() as cur:
            for i, c in enumerate(chunks):
                cur.execute(
                    """INSERT INTO chunks (chunk_id, ticker, payload, embedding)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (chunk_id) DO UPDATE
                       SET payload = EXCLUDED.payload, embedding = EXCLUDED.embedding""",
                    (c.chunk_id, c.ticker, c.model_dump_json(), embeddings[i]),
                )

    def all_chunks(self, ticker: str | None = None) -> list[Chunk]:
        q = "SELECT payload FROM chunks" + (" WHERE ticker = %s" if ticker else "")
        params = (ticker.upper(),) if ticker else ()
        return [Chunk(**row[0]) for row in self._conn.execute(q, params).fetchall()]

    def search(self, query_vec: np.ndarray, top_k: int = 8, ticker: str | None = None) -> list[tuple[Chunk, float]]:
        where = "WHERE ticker = %(ticker)s" if ticker else ""
        rows = self._conn.execute(
            f"""SELECT payload, 1 - (embedding <=> %(vec)s) AS score
                FROM chunks {where}
                ORDER BY embedding <=> %(vec)s LIMIT %(k)s""",
            {"vec": query_vec, "k": top_k, "ticker": (ticker or "").upper()},
        ).fetchall()
        return [(Chunk(**r[0]), float(r[1])) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def get_store(dim: int):
    from ..config import get_settings

    s = get_settings()
    if s.postgres_dsn:
        return PgVectorStore(s.postgres_dsn, dim)
    return SqliteStore(s.db_path)
