from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source filing (or filing section) registered for a ticker."""

    doc_id: str
    ticker: str
    form_type: str  # e.g. 10-K, 10-Q, 8-K
    section: str  # e.g. "Item 7. MD&A"
    period: str  # e.g. "FY2025"
    text: str
    source_url: str = ""


class Chunk(BaseModel):
    """A retrievable unit with enough metadata to render a citation."""

    chunk_id: str
    doc_id: str
    ticker: str
    form_type: str
    section: str
    period: str
    text: str
    char_start: int
    char_end: int


class RetrievalHit(BaseModel):
    chunk: Chunk
    score: float
    source: str  # "bm25" | "dense" | "fused"


class Citation(BaseModel):
    chunk_id: str
    ticker: str
    form_type: str
    section: str
    period: str
    quote: str = ""


class VerifiedClaim(BaseModel):
    claim: str
    citation: Citation
    supported: bool
    evidence: str = ""


class AgentEvent(BaseModel):
    """One step of the pipeline trace, streamed to clients over SSE."""

    node: str
    status: str  # "started" | "finished"
    detail: dict = Field(default_factory=dict)


class Answer(BaseModel):
    question: str
    route: str
    answer: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    model_used: str = ""
    unsupported_claims_removed: int = 0
