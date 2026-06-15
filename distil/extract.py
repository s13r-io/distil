"""Stage 2 — Extraction, routed by knowledge type. ARCHITECTURE.md §2; TESTING T-E1..E4.

The triage verdict's dominant type selects a type-specific prompt (heuristic keeps rationale +
scope; procedural keeps order). Parsing and quote discipline are deterministic and unit-tested;
faithfulness of the model's output is the gated eval (T-E3).

**Quote discipline (T-E4)** is enforced here in code: any item whose ``provenance.quote`` is
15 words or longer is rejected outright — a copyright/faithfulness guardrail that does not
depend on the model behaving.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .ingest import Transcript
from .llm import LLMClient
from .models import KnowledgeItem, Triage
from .prompts.extract import SYSTEM, build_extract_prompt
from .triage import ParseError

_MAX_QUOTE_WORDS = 15
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class QuoteDisciplineError(ValueError):
    """Raised when an extracted item's provenance quote violates the <15-word rule (T-E4)."""


def dominant_type(triage: Triage) -> str:
    """The knowledge type with the largest share; defaults to 'conceptual' if none given."""
    if not triage.knowledge_types_present:
        return "conceptual"
    return max(triage.knowledge_types_present, key=lambda kt: kt.share).type


def run_extraction(
    transcript: Transcript, triage: Triage, client: LLMClient
) -> list[KnowledgeItem]:
    ktype = dominant_type(triage)
    prompt = build_extract_prompt(ktype, transcript.full_text())
    raw = client.complete(prompt, system=SYSTEM)
    items = _parse_items(raw)
    _enforce_quote_discipline(items)
    return items


def _parse_items(raw: str) -> list[KnowledgeItem]:
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ParseError(f"Extraction response was not a JSON array: {raw[:120]!r}") from exc
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            raise ParseError(
                f"Extraction response was not a JSON array: {raw[:120]!r}"
            ) from exc2
    if not isinstance(data, list):
        raise ParseError("Extraction response must be a JSON array of items.")

    items: list[KnowledgeItem] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ParseError("Each extracted item must be a JSON object.")
        obj.setdefault("item_id", f"k_{i + 1:02d}")
        try:
            items.append(KnowledgeItem.model_validate(obj))
        except ValidationError as exc:
            raise ParseError(f"Extracted item {i} did not match the schema: {exc}") from exc
    return items


def _enforce_quote_discipline(items: list[KnowledgeItem]) -> None:
    for item in items:
        word_count = len(item.provenance.quote.split())
        if word_count >= _MAX_QUOTE_WORDS:
            raise QuoteDisciplineError(
                f"Provenance quote has {word_count} words (limit {_MAX_QUOTE_WORDS - 1}): "
                f"{item.provenance.quote!r}"
            )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped
