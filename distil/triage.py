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

**Not called by the production pipeline anymore.** ``run_pipeline`` reads the transcript once,
through ``extract.run_triage_extract`` (a merged triage+extraction call — see that function's
docstring and ``distil/prompts/triage_extract.py``), which reuses this module's JSON parser
(:func:`parse_triage_response`) on its ``<TRIAGE>`` section. ``run_triage``/this module's own
standalone model call stay as a reusable, independently-testable classifier — the gated eval
suite (``tests/eval/test_triage_eval.py``) and any future diagnostic use still exercise it
directly, isolated from extraction.
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
    """Parse a raw model response into a schema-valid :class:`Triage`.

    Shared with ``extract.run_triage_extract``, which parses this exact JSON shape out of a
    larger merged response's ``<TRIAGE>`` section — the format (fenced or bare JSON object,
    optionally with surrounding prose) is identical either way.
    """
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
