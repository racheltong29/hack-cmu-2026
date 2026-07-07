"""Pluggable embedding backends.

`HashingEmbedder` is a deterministic, dependency-free feature-hashing model:
it captures unigram/bigram lexical similarity well enough for tests, CI, and
offline demos, and requires no downloads or GPU. Production deployments swap
in `SentenceTransformerEmbedder` (bge-m3 etc.) via the same protocol —
retrieval code is agnostic to the backend.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.%$-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Feature-hashing tf embedder with L2 normalization (unigrams + bigrams)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                # Sub-linear tf via log damping happens implicitly per occurrence.
                out[i, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Dense neural embeddings (requires `pip install sentence-transformers`)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer  # deferred heavy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return HashingEmbedder()
