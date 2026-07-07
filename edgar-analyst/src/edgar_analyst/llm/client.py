"""LLM backends.

`AnthropicLLM` is the production client (Claude via the official SDK).
`StubLLM` is a deterministic extractive backend so the full pipeline —
routing, analysis, verification, synthesis — runs hermetically in tests,
CI, and keyless demos. Both satisfy the same two-method protocol, so the
agent graph never knows which one it's talking to.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from ..config import get_settings
from ..retrieval.embeddings import tokenize


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, model: str, max_tokens: int = 2048) -> str: ...

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict: ...


class AnthropicLLM:
    name = "anthropic"

    def __init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str, model: str, max_tokens: int = 2048) -> str:
        kwargs: dict = {}
        # Haiku doesn't support adaptive thinking; Opus-tier models do.
        if "haiku" not in model:
            kwargs["thinking"] = {"type": "adaptive"}
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in message.content if b.type == "text")

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict:
        response = self._client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            return {}
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(text)


class StubLLM:
    """Extractive, rule-based stand-in for offline runs.

    Routing is keyword-based; analysis selects evidence sentences that
    overlap the question; synthesis templates verified claims. Deterministic
    by construction, so eval metrics are stable across runs.
    """

    name = "stub"

    def complete(self, system: str, prompt: str, model: str, max_tokens: int = 2048) -> str:
        if "TASK: route" in prompt:
            return self._route(prompt)
        if "TASK: synthesize" in prompt:
            return self._synthesize(prompt)
        return "Unsupported stub task."

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict:
        if "TASK: extract_claims" in prompt:
            return self._extract_claims(prompt)
        return {}

    # -- routing ---------------------------------------------------------
    @staticmethod
    def _route(prompt: str) -> str:
        q = prompt.lower()
        if any(w in q for w in ("risk", "headwind", "concern", "threat", "exposure")):
            return "risk"
        if any(w in q for w in (" vs ", "versus", "compare", "both companies", "which company")):
            return "comparison"
        return "filings"

    _STOPWORDS = frozenset(
        "a an and are as at be by did do does for from has have how in is it its "
        "of on or that the this to was were what when which who why will with".split()
    )

    @staticmethod
    def _stem(token: str) -> str:
        if len(token) > 4:
            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix):
                    token = token[: -len(suffix)]
                    break
            token = token.rstrip("e")
        return token

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        return {cls._stem(t) for t in tokenize(text) if t not in cls._STOPWORDS}

    # -- claim extraction --------------------------------------------------
    @classmethod
    def _extract_claims(cls, prompt: str) -> dict:
        question_match = re.search(r"QUESTION:\s*(.+)", prompt)
        question = question_match.group(1) if question_match else ""
        q_tokens = cls._content_tokens(question)

        claims = []
        # Evidence blocks are rendered as: [chunk_id] text...
        for chunk_id, text in re.findall(r"\[([^\]\n]+)\]\n(.+?)(?=\n\[|\Z)", prompt, re.S):
            scored: list[tuple[float, str]] = []
            for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
                overlap = len(q_tokens & cls._content_tokens(sent))
                if overlap < 2:  # require at least two shared content words
                    continue
                score = overlap / (len(q_tokens) or 1)
                # Prefer sentences carrying figures — they make checkable claims.
                if re.search(r"\d", sent):
                    score += 0.15
                scored.append((score, sent.strip()))
            scored.sort(key=lambda t: -t[0])
            for score, sent in scored[:2]:
                if score >= 0.3:
                    claims.append({"claim": sent, "chunk_id": chunk_id.strip()})
        return {"claims": claims[:6]}

    # -- synthesis ---------------------------------------------------------
    @staticmethod
    def _synthesize(prompt: str) -> str:
        claims = re.findall(r"CLAIM \[(\d+)\]:\s*(.+)", prompt)
        if not claims:
            return "No sufficiently supported evidence was found in the indexed filings to answer this question."
        lines = [f"{text} [{num}]" for num, text in claims]
        return " ".join(lines)


def get_llm() -> LLMClient:
    provider = get_settings().llm_provider
    if provider == "stub":
        return StubLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    # auto: use Anthropic when credentials resolve, otherwise run offline.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return AnthropicLLM()
    return StubLLM()
