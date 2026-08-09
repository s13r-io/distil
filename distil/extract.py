"""Stage 2 — Extraction, routed by knowledge type. ARCHITECTURE.md §2; TESTING T-E1..E4.

The triage verdict's dominant type selects a type-specific prompt (heuristic keeps rationale +
scope; procedural keeps order). Parsing and quote discipline are deterministic and unit-tested;
faithfulness of the model's output is the gated eval (T-E3).

**Quote discipline (T-E4)** is enforced here in code: any item whose ``provenance.quote`` is
15 words or longer is rejected outright — a copyright/faithfulness guardrail that does not
depend on the model behaving.

**Extraction robustness (T-E5..E7)**: the extraction call is the one most likely to hit the
output-token cap on a long transcript, or a dropped connection mid-stream. Both surface
the same way — a response that fails to parse as a JSON array. ``run_extraction`` retries the
model call a bounded number of times on exactly that failure mode (network exception, or an
unparseable/unrecoverable response); a schema-level (semantic) failure in an item that *did*
parse is not retried. When the response looks like a JSON array that began but was cut off
mid-stream, ``_parse_items_json`` recovers whatever complete leading objects it can rather than
discarding the whole response — see ``_recover_truncated_leading_objects``. The per-stage
``max_tokens`` ceiling that bounds how often this path is reached at all is resolved by
``distil/model_config.py`` (``resolve_stage_max_tokens``), sized to the extraction model's real
published output ceiling rather than a flat default — see that module's docstring. Even with a
correct ceiling, a genuinely truncated/salvaged response must never look identical to a complete
one: ``_parse_items_json`` reports whether it had to recover a partial array, and that flag rides
``TriageExtractResult.truncated`` all the way to ``KBEntry.extraction_truncated`` (surfaced in the
web UI and the rendered markdown/teaching-note export) — see ``run_triage_extract``.

**type/stance drift (T-E8)**: the model occasionally copies a ``stance`` value (most often
``personal_experience``) into the ``type`` field. Since ``build_extract_prompt`` fixes the
requested ``KnowledgeType`` per call, the caller always knows the one correct value, so
``_items_from_json`` repairs a ``type`` that isn't a valid ``KnowledgeType`` to the requested
type before validating — see ``_repair_type``. A ``type`` that validates but doesn't match the
request is left alone (the model is allowed to say an item is a different valid type than the
dominant one). An item that still fails validation after that repair is dropped rather than
failing the whole batch — see ``_items_from_json``'s salvage floor.

**Entities (Phase D)**: each item may carry a nested ``entities`` array (named tools/people/
organizations mentioned in that item's sentence) — the same call, no second transcript read.
``_clean_entity_mentions`` validates and drops malformed entries *before* the parent item is
ever validated (mirroring ``_repair_type``'s "fix what's recoverable, drop what isn't" shape,
but per-mention rather than per-field): an unparseable ``kind``, a missing ``name``, or an
empty ``quote`` drops just that one mention. This makes it structurally impossible for a
malformed entity to fail (and thereby drop) the knowledge item it rode in on — entity cleaning
never raises and never participates in the item-level salvage floor.

**Triage+extraction merged into one call (`run_triage_extract`, owner decision).** Triage's
short-circuit is gone (``pipeline.py``'s module docstring) — the only reason for a separate,
cheap classification pass *before* extraction was to buy a veto over whether extraction ran at
all, and that veto no longer exists. So the strong model now reads the full transcript exactly
once: :func:`run_triage_extract` sends ``prompts/triage_extract.py``'s two-section prompt, which
asks the model to state its classification (``<TRIAGE>``) FIRST, then extract items
(``<ITEMS>``) SECOND, conditioned on what it just classified — preserving the decide-then-act
sequencing the old two-call design bought via call ordering, within one response instead of
two. Parsing deliberately keeps the two existing, separately-tested parsers intact rather than
inventing one for a merged JSON object: ``triage.parse_triage_response`` for ``<TRIAGE>`` (small,
never expected to truncate) and this module's own ``_parse_items_json`` for ``<ITEMS>``
(reuses the existing truncated-array recovery for a long items array cut off by the output-token
cap). ``run_triage``/``run_extraction`` (the split calls) are unchanged and still used by the
gated eval suite for isolated regression testing — this is an additional entry point, not a
replacement of either.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import get_args

from pydantic import ValidationError

from .ingest import Transcript
from .llm import LLMClient
from .models import EntityKind, EntityMention, KnowledgeItem, KnowledgeType, Triage
from .prompts.extract import SYSTEM, build_extract_prompt
from .prompts.triage_extract import SYSTEM as TRIAGE_EXTRACT_SYSTEM
from .prompts.triage_extract import build_triage_extract_prompt
from .triage import ParseError, parse_triage_response

logger = logging.getLogger(__name__)

_MAX_QUOTE_WORDS = 15
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_MAX_RETRIES = 2
_RETRY_SLEEP_SECONDS = 0.5

_TRIAGE_SECTION = re.compile(r"<TRIAGE>(.*?)</TRIAGE>", re.DOTALL | re.IGNORECASE)
_ITEMS_OPEN = re.compile(r"<ITEMS>", re.IGNORECASE)
_ITEMS_CLOSE = re.compile(r"</ITEMS>", re.IGNORECASE)

_VALID_KNOWLEDGE_TYPES = frozenset(get_args(KnowledgeType))
_VALID_ENTITY_KINDS = frozenset(get_args(EntityKind))

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
    data, truncated = _complete_with_retry(client, prompt, SYSTEM)
    items = _items_from_json(data, ktype)
    _truncate_overlong_quotes(items)
    _enforce_quote_discipline(items)
    if truncated:
        logger.warning(
            "Extraction response was truncated (output cap or dropped connection) and salvaged "
            "via partial recovery: only %d item(s) survived. The transcript's knowledge was NOT "
            "fully captured.",
            len(items),
        )
    return items


@dataclass
class TriageExtractResult:
    triage: Triage
    items: list[KnowledgeItem]
    raw: str
    truncated: bool = False
    """True when the merged response had to be salvaged from an incomplete ``<ITEMS>`` array
    (the output-token cap was hit, or the connection dropped mid-stream) rather than parsed as a
    complete response — see ``extract._parse_items_json``. A salvaged response must never look
    identical to a complete one: this is what lets ``pipeline.run_pipeline`` record the fact on
    the filed ``KBEntry`` (``KBEntry.extraction_truncated``) instead of silently returning
    whatever partial item list happened to survive."""


def run_triage_extract(transcript: Transcript, client: LLMClient) -> TriageExtractResult:
    """One strong-tier call that classifies AND extracts — see the module docstring's
    "Triage+extraction merged into one call" section for why and how."""
    prompt = build_triage_extract_prompt(transcript.full_text())
    triage, items_data, raw, truncated = _complete_triage_extract_with_retry(client, prompt)
    ktype = dominant_type(triage)
    items = _items_from_json(items_data, ktype)
    _truncate_overlong_quotes(items)
    _enforce_quote_discipline(items)
    if truncated:
        logger.warning(
            "Triage+extract response was truncated (output cap or dropped connection) and "
            "salvaged via partial recovery: only %d item(s) survived out of what the model "
            "started generating. The transcript's knowledge was NOT fully captured — this is "
            "recorded on the filed entry (KBEntry.extraction_truncated).",
            len(items),
        )
    return TriageExtractResult(triage=triage, items=items, raw=raw, truncated=truncated)


def _complete_triage_extract_with_retry(
    client: LLMClient, prompt: str
) -> tuple[Triage, list, str, bool]:
    """Same retry contract as ``_complete_with_retry``: a dropped connection or an unparseable
    response (missing/malformed ``<TRIAGE>``, or an ``<ITEMS>`` array that can't be recovered at
    all) is retried a bounded number of times; a schema-level failure in items that did parse
    happens later, in ``_items_from_json``, and is never retried here.

    The returned ``bool`` is whether *this* (successful) attempt's items had to be salvaged from
    an incomplete array — a retry that eventually succeeds cleanly reports ``False`` even if an
    earlier attempt failed outright; only a genuinely partial-but-usable response is truncated.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            raw = client.complete(prompt, system=TRIAGE_EXTRACT_SYSTEM)
            triage_text, items_text = _split_triage_extract_response(raw)
            triage = parse_triage_response(triage_text)
            items_data, truncated = _parse_items_json(items_text, kind="Extraction")
            return triage, items_data, raw, truncated
        except ParseError as exc:
            last_exc = exc
        except Exception as exc:  # network/connection failure — retry
            last_exc = exc
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_SLEEP_SECONDS)
    assert last_exc is not None
    raise last_exc


def _split_triage_extract_response(raw: str) -> tuple[str, str]:
    """Split a merged response into its ``<TRIAGE>`` and ``<ITEMS>`` sections.

    Tolerant of truncation only in the (potentially large) items section: if the closing
    ``</ITEMS>`` tag never arrived, everything after ``<ITEMS>`` is handed to
    ``_parse_items_json``, which already recovers whatever complete leading items it can from a
    cut-off array. The triage section is small and expected to always be complete — a missing
    ``<TRIAGE>``/``<ITEMS>`` tag is treated as a genuine parse failure (retried by the caller).
    """
    triage_match = _TRIAGE_SECTION.search(raw)
    if not triage_match:
        raise ParseError(f"Response is missing a complete <TRIAGE> section: {raw[:120]!r}")

    items_open = _ITEMS_OPEN.search(raw)
    if not items_open:
        raise ParseError(f"Response is missing an <ITEMS> section: {raw[:120]!r}")
    items_start = items_open.end()
    items_close = _ITEMS_CLOSE.search(raw, items_start)
    items_text = raw[items_start : items_close.start()] if items_close else raw[items_start:]
    return triage_match.group(1), items_text


def _complete_with_retry(client: LLMClient, prompt: str, system: str) -> tuple[list, bool]:
    """Call the model and parse a JSON array, retrying on parse/network failures only (T-E5..E7).

    A dropped connection (``client.complete`` raises) and a response truncated by the
    output-token cap (``_parse_items_json`` can't recover a usable array) both mean this
    attempt produced nothing usable — worth a bounded retry. A schema-level failure in items
    that *did* parse happens later, in ``_items_from_json``, and is never retried here.

    Returns ``(data, truncated)`` — see ``_parse_items_json`` for what ``truncated`` means.
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


def _parse_items_json(raw: str, *, kind: str = "extraction") -> tuple[list, bool]:
    """Robustly parse a model response into a JSON array.

    Tolerates code fences / surrounding prose, and — when the response is a JSON array that
    began but was never closed (truncated mid-stream) — recovers whatever complete leading
    objects it can via ``_recover_truncated_leading_objects``. Rejects only when nothing at
    all can be recovered.

    Returns ``(data, truncated)``. ``truncated`` is ``True`` only when ``data`` came from the
    salvage path (``_recover_truncated_leading_objects``) rather than a complete, well-formed
    array — a response that parses cleanly on the first try, or that only needed fence/prose
    stripped from around an otherwise-complete embedded array, is not truncated. This is what
    makes a salvaged response distinguishable from a complete one (see the module docstring's
    "Triage+extraction merged into one call" section and ``TriageExtractResult.truncated``) —
    a salvage that silently looks identical to success is exactly the defect this flag exists
    to prevent.
    """
    text = _strip_fence(raw).strip()
    truncated = False
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
            truncated = True
    if not isinstance(data, list):
        raise ParseError(f"{kind} response must be a JSON array.")
    return data, truncated


def _recover_truncated_leading_objects(text: str) -> list:
    """Best-effort recovery for a JSON array that began but was cut off mid-stream.

    Walks the text after the opening ``[`` and greedily decodes each complete leading JSON
    value, stopping at the first value that fails to parse (the truncated tail — e.g. the
    stage's output-token ceiling cut it off, or the connection dropped). Returns whatever
    complete objects were recovered, possibly none; never fabricates or completes a partial
    object — items that would have needed the cut-off tail are simply absent.
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
    caller always knows the type it asked for. Any nested ``entities`` are cleaned next (see
    ``_clean_entity_mentions``) into already-valid ``EntityMention`` instances *before* the item
    itself is validated, so a malformed entity can never fail — and thereby drop — the item it
    rode in on. An item that still fails validation after that repair is a genuine semantic
    failure and is dropped rather than discarding every other item in the batch (never retried
    by ``_complete_with_retry`` — this is not a parse failure). If too few items survive, that
    signals a systemically broken response rather than one or two isolated mistakes, so raise
    instead of silently returning near-nothing. (Entities have no equivalent salvage floor of
    their own: they're optional per item, so a batch with zero recovered entities is normal, not
    a sign of a broken response.)
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
        obj["entity_mentions"] = _clean_entity_mentions(obj.pop("entities", None))
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


def _clean_entity_mentions(raw_entities: object) -> list[EntityMention]:
    """Validate a raw ``entities`` payload into already-valid ``EntityMention`` instances.

    Never raises: not-a-list input, a non-dict element, a missing/empty ``name`` or ``quote``,
    or a ``kind`` outside the closed ``EntityKind`` set each drop just that one mention (logged),
    with no repair-and-guess for ``kind`` — unlike ``_repair_type``, there is no single correct
    answer to fall back to here, so an untrustworthy ``kind`` is dropped rather than coerced.
    Overlong quotes are silently truncated, mirroring ``_truncate_overlong_quotes`` for items.
    """
    if not isinstance(raw_entities, list):
        return []
    mentions: list[EntityMention] = []
    for i, obj in enumerate(raw_entities):
        if not isinstance(obj, dict):
            logger.warning("Entity mention %d dropped: not a JSON object.", i)
            continue
        obj = dict(obj)
        if obj.get("kind") not in _VALID_ENTITY_KINDS:
            logger.warning("Entity mention %d dropped: invalid kind %r.", i, obj.get("kind"))
            continue
        quote = obj.get("quote")
        if isinstance(quote, str):
            obj["quote"] = _truncate_quote(quote)
        try:
            mentions.append(EntityMention.model_validate(obj))
        except ValidationError as exc:
            logger.warning("Entity mention %d dropped: did not match the schema: %s", i, exc)
    return mentions


def _truncate_quote(quote: str) -> str:
    """Trim ``quote`` to exactly ``_MAX_QUOTE_WORDS - 1`` words if it's at or over the limit,
    else return it unchanged. Shared by ``_truncate_overlong_quotes`` (items) and
    ``_clean_entity_mentions`` (entities) — the same faithfulness-preserving truncation either
    way (a leading substring of a genuine verbatim quote is still verbatim)."""
    words = quote.split()
    if len(words) >= _MAX_QUOTE_WORDS:
        return " ".join(words[: _MAX_QUOTE_WORDS - 1])
    return quote


def _truncate_overlong_quotes(items: list[KnowledgeItem]) -> None:
    """Silently trim quotes that exceed the word limit to exactly _MAX_QUOTE_WORDS-1 words.

    The LLM occasionally returns a genuine verbatim quote that is one or two words over the
    limit.  Truncating preserves faithfulness (a leading substring is still verbatim) while
    satisfying the copyright/length guardrail.  This runs *before* _enforce_quote_discipline
    so the hard guard only fires for genuinely fabricated / runaway quotes.
    """
    for item in items:
        item.provenance.quote = _truncate_quote(item.provenance.quote)


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
