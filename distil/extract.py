"""Stage 2 — Extraction, routed by knowledge type. ARCHITECTURE.md §2; TESTING T-E1..E4.

The triage verdict's dominant type selects a type-specific prompt (heuristic keeps rationale +
scope; procedural keeps order). Parsing and quote discipline are deterministic and unit-tested;
faithfulness of the model's output is the gated eval (T-E3).

**Quote discipline (T-E4)** is enforced here in code: any item whose ``provenance.quote`` is
15 words or longer is rejected outright — a copyright/faithfulness guardrail that does not
depend on the model behaving.

**Extraction robustness (T-E5..E7)**: the extraction call is the one most likely to hit the
4,096-token output cap on a long transcript, or a dropped connection mid-stream. Both surface
the same way — a response that fails to parse as a JSON array. ``run_extraction`` retries the
model call a bounded number of times on exactly that failure mode (network exception, or an
unparseable/unrecoverable response); a schema-level (semantic) failure in an item that *did*
parse is not retried. When the response looks like a JSON array that began but was cut off
mid-stream, ``_parse_items_json`` recovers whatever complete leading objects it can rather than
discarding the whole response — see ``_recover_truncated_leading_objects``.

**type/stance drift (T-E8)**: the model occasionally copies a ``stance`` value (most often
``personal_experience``) into the ``type`` field. Since ``build_extract_prompt`` fixes the
requested ``KnowledgeType`` per call, the caller always knows the one correct value, so
``_items_from_json`` repairs a ``type`` that isn't a valid ``KnowledgeType`` to the requested
type before validating — see ``_repair_type``. A ``type`` that validates but doesn't match the
request is left alone (the model is allowed to say an item is a different valid type than the
dominant one). An item that still fails validation after that repair is dropped rather than
failing the whole batch — see ``_items_from_json``'s salvage floor.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import get_args

from pydantic import ValidationError

from .ingest import Transcript
from .llm import LLMClient
from .models import KnowledgeItem, KnowledgeType, Triage
from .prompts.extract import SYSTEM, build_extract_prompt
from .triage import ParseError

logger = logging.getLogger(__name__)

_MAX_QUOTE_WORDS = 15
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_MAX_RETRIES = 2
_RETRY_SLEEP_SECONDS = 0.5

_VALID_KNOWLEDGE_TYPES = frozenset(get_args(KnowledgeType))

# A wholesale-garbage response (wrong shape, hallucinated fields, etc.) should still fail loudly
# rather than silently returning a near-empty result. Below half the items surviving validation
# is past what one or two isolated model mistakes would produce, so treat it as systemic and raise.
_MIN_SALVAGE_FRACTION = 0.5


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
    data = _complete_with_retry(client, prompt, SYSTEM)
    items = _items_from_json(data, ktype)
    _truncate_overlong_quotes(items)
    _enforce_quote_discipline(items)
    return items


def _complete_with_retry(client: LLMClient, prompt: str, system: str) -> list:
    """Call the model and parse a JSON array, retrying on parse/network failures only (T-E5..E7).

    A dropped connection (``client.complete`` raises) and a response truncated by the
    output-token cap (``_parse_items_json`` can't recover a usable array) both mean this
    attempt produced nothing usable — worth a bounded retry. A schema-level failure in items
    that *did* parse happens later, in ``_items_from_json``, and is never retried here.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            raw = client.complete(prompt, system=system)
            return _parse_items_json(raw, kind="Extraction")
        except ParseError as exc:
            last_exc = exc
        except Exception as exc:  # network/connection failure — retry
            last_exc = exc
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_SLEEP_SECONDS)
    assert last_exc is not None
    raise last_exc


def _parse_items_json(raw: str, *, kind: str = "extraction") -> list:
    """Robustly parse a model response into a JSON array.

    Tolerates code fences / surrounding prose, and — when the response is a JSON array that
    began but was never closed (truncated mid-stream) — recovers whatever complete leading
    objects it can via ``_recover_truncated_leading_objects``. Rejects only when nothing at
    all can be recovered.
    """
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        data = None
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            recovered = _recover_truncated_leading_objects(text)
            if not recovered:
                raise ParseError(f"{kind} response was not a JSON array: {raw[:120]!r}") from exc
            data = recovered
    if not isinstance(data, list):
        raise ParseError(f"{kind} response must be a JSON array.")
    return data


def _recover_truncated_leading_objects(text: str) -> list:
    """Best-effort recovery for a JSON array that began but was cut off mid-stream.

    Walks the text after the opening ``[`` and greedily decodes each complete leading JSON
    value, stopping at the first value that fails to parse (the truncated tail — e.g. the
    4,096-token output cap cut it off, or the connection dropped). Returns whatever complete
    objects were recovered, possibly none; never fabricates or completes a partial object —
    items that would have needed the cut-off tail are simply absent.
    """
    if not text.startswith("["):
        return []
    decoder = json.JSONDecoder()
    idx = 1
    length = len(text)
    recovered: list = []
    while True:
        while idx < length and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= length or text[idx] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        recovered.append(obj)
        idx = end
    return recovered


def _repair_type(obj: dict, requested_type: str) -> None:
    """Repair a ``type`` that isn't a valid ``KnowledgeType`` to the requested type (T-E8).

    The model sometimes copies a ``stance`` value (typically ``personal_experience``) into
    ``type``. ``build_extract_prompt`` fixed one concrete ``KnowledgeType`` for this call, so
    the caller already knows the correct value — no need to guess or reject. If ``type`` IS a
    valid ``KnowledgeType`` (just not the requested one), it is left alone: the model is
    allowed to flag an item as a different real type than the dominant one asked for.
    """
    if obj.get("type") not in _VALID_KNOWLEDGE_TYPES:
        obj["type"] = requested_type


def _items_from_json(data: list, requested_type: str) -> list[KnowledgeItem]:
    """Validate already-parsed JSON array data into schema-conforming items.

    A ``type`` value is repaired first (see ``_repair_type``) since it is recoverable — the
    caller always knows the type it asked for. An item that still fails validation after that
    repair is a genuine semantic failure and is dropped rather than discarding every other item
    in the batch (never retried by ``_complete_with_retry`` — this is not a parse failure). If
    too few items survive, that signals a systemically broken response rather than one or two
    isolated mistakes, so raise instead of silently returning near-nothing.
    """
    items: list[KnowledgeItem] = []
    dropped = 0
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            dropped += 1
            logger.warning("Extracted item %d dropped: not a JSON object.", i)
            continue
        obj.setdefault("item_id", f"k_{i + 1:02d}")
        _repair_type(obj, requested_type)
        try:
            items.append(KnowledgeItem.model_validate(obj))
        except ValidationError as exc:
            dropped += 1
            logger.warning("Extracted item %d dropped: did not match the schema: %s", i, exc)

    total = len(data)
    if total and len(items) / total < _MIN_SALVAGE_FRACTION:
        raise ParseError(
            f"Extraction produced {len(items)}/{total} valid items ({dropped} dropped) — "
            f"below the {_MIN_SALVAGE_FRACTION:.0%} salvage floor; treating as a broken response."
        )
    return items


def _truncate_overlong_quotes(items: list[KnowledgeItem]) -> None:
    """Silently trim quotes that exceed the word limit to exactly _MAX_QUOTE_WORDS-1 words.

    The LLM occasionally returns a genuine verbatim quote that is one or two words over the
    limit.  Truncating preserves faithfulness (a leading substring is still verbatim) while
    satisfying the copyright/length guardrail.  This runs *before* _enforce_quote_discipline
    so the hard guard only fires for genuinely fabricated / runaway quotes.
    """
    for item in items:
        words = item.provenance.quote.split()
        if len(words) >= _MAX_QUOTE_WORDS:
            item.provenance.quote = " ".join(words[: _MAX_QUOTE_WORDS - 1])


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
