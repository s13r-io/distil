"""Stage 1 — Triage. ARCHITECTURE.md §2; TESTING T-T1..T5.

Classifies a transcript (dominant knowledge type, density, transcript-loss). The deterministic
glue here — prompt assembly, JSON extraction, schema validation — is unit-tested against a
``FakeClient``; the model's judgment is checked by the gated eval suite.

The ``knowledge_types_present`` classification drives what ``extract.py`` looks for and is the
quality-critical output; ``density``/``transcript_loss`` are informational context for the note.
``verdict`` is still produced (the schema/prompt/rendering shape other code and stored data
depend on), but the pipeline never acts on it — the owner's decision to keep a video is trusted
unconditionally, and the only quality gate is ``ingest.py``'s word-count check (owner decision;
see ``distil/pipeline.py``'s module docstring).

**Runs once per pipeline call, on the cheap tier (owner decision).** ``run_pipeline`` calls
:func:`run_triage` first, over the whole transcript; its dominant type then steers
``extract.run_chunked_extraction`` (strong tier). This is a coarse categorical judgement, not
extraction's faithfulness-critical verbatim-quote work, so the cheap tier is the measured,
justified default here (see ``model_config.py``'s tier assignment and the PR that set it) — it
was briefly merged into one strong-tier call alongside extraction, then split back out because
chunked extraction (several per-chunk calls) cannot produce one whole-transcript classification
without disagreeing verdicts across chunks.
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
    triage = parse_triage_response(raw)
    return TriageResult(triage=triage, raw=raw)


def parse_triage_response(raw: str) -> Triage:
    """Parse a raw model response into a schema-valid :class:`Triage` (fenced or bare JSON
    object, optionally with surrounding prose)."""
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
