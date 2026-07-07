"""Sentence-aware sliding-window chunker.

Splits on sentence boundaries so a chunk never cuts a number off from its
context ("revenue was $20.1" / "billion"), which matters for citation quality.
"""

from __future__ import annotations

import re

from ..config import get_settings
from ..models import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_document(doc: Document, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    s = get_settings()
    size = size or s.chunk_size_chars
    overlap = overlap if overlap is not None else s.chunk_overlap_chars

    sentences = split_sentences(doc.text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    char_cursor = 0
    chunk_start = 0

    def flush() -> None:
        nonlocal buf, buf_len, chunk_start
        if not buf:
            return
        text = " ".join(buf)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                ticker=doc.ticker,
                form_type=doc.form_type,
                section=doc.section,
                period=doc.period,
                text=text,
                char_start=chunk_start,
                char_end=chunk_start + len(text),
            )
        )

    for sent in sentences:
        if buf_len + len(sent) > size and buf:
            flush()
            # Carry trailing sentences forward as overlap for context continuity.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(buf):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev)
            chunk_start = char_cursor - carried_len
            buf = carried
            buf_len = carried_len
        buf.append(sent)
        buf_len += len(sent) + 1
        char_cursor += len(sent) + 1
    flush()
    return chunks
