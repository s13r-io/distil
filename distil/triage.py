"""Stage 1 — Triage. ARCHITECTURE.md §2; TESTING T-T1..T5.

Classifies a transcript (types present, density, transcript-loss, verdict). The deterministic
glue here — prompt assembly, JSON extraction, schema validation — is unit-tested against a
``FakeClient``; the model's judgment is checked by the gated eval suite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .ingest import Transcript
from .llm import LLMClient
from .models import Triage
from .prompts.triage import SYSTEM, build_triage_prompt

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ParseError(ValueError):
    """Raised when the model response is not valid, schema-conforming triage JSON."""


@dataclass
class TriageResult:
    triage: Triage
    raw: str


def run_triage(transcript: Transcript, client: LLMClient) -> TriageResult:
    prompt = build_triage_prompt(transcript.full_text())
    raw = client.complete(prompt, system=SYSTEM)
    triage = _parse(raw)
    return TriageResult(triage=triage, raw=raw)


def is_low_value(result: TriageResult) -> bool:
    """The honesty short-circuit signal: pipeline must not extract when this is True (T-T3)."""
    return result.triage.verdict == "little_to_extract"


def _parse(raw: str) -> Triage:
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Last resort: pull the first {...} block out of surrounding prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ParseError(f"Triage response was not JSON: {raw[:120]!r}") from exc
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            raise ParseError(f"Triage response was not JSON: {raw[:120]!r}") from exc2
    try:
        return Triage.model_validate(data)
    except ValidationError as exc:
        raise ParseError(f"Triage JSON did not match the schema: {exc}") from exc


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped
