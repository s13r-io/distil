"""Stage 2 — Extraction, routed by knowledge type. ARCHITECTURE.md §2; TESTING T-E1..E4.

The triage verdict's dominant type selects a type-specific prompt (heuristic keeps rationale +
scope; procedural keeps order). Parsing and quote discipline are deterministic and unit-tested;
faithfulness of the model's output is the gated eval (T-E3).

**Quote discipline (T-E4)** is enforced here in code: any item whose ``provenance.quote`` is
15 words or longer is rejected outright — a copyright/faithfulness guardrail that does not
depend on the model behaving.

**Extraction robustness (T-E5..E7)**: the extraction call is the one most likely to hit the
output-token cap on a long transcript (or a chunk of one), or a dropped connection mid-stream.
Both surface the same way — a response that fails to parse as a JSON array. ``run_extraction``
and each per-chunk call inside ``run_chunked_extraction`` retry the model call a bounded number
of times on exactly that failure mode (network exception, or an unparseable/unrecoverable
response); a schema-level (semantic) failure in an item that *did* parse is not retried. When the
response looks like a JSON array that began but was cut off mid-stream, ``_parse_items_json``
recovers whatever complete leading objects it can rather than discarding the whole response —
see ``_recover_truncated_leading_objects``. The per-stage ``max_tokens`` ceiling that bounds how
often this path is reached at all is resolved by ``distil/model_config.py``
(``resolve_stage_max_tokens``), sized to the extraction model's real published output ceiling
rather than a flat default — see that module's docstring. Even with a correct ceiling, a
genuinely truncated/salvaged response must never look identical to a complete one:
``_parse_items_json`` reports whether it had to recover a partial array, and that flag rides
``ChunkedExtractionResult.truncated`` all the way to ``KBEntry.extraction_truncated`` (surfaced
in the web UI and the rendered markdown/teaching-note export) — see ``run_chunked_extraction``.

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

**Classification and extraction are separate calls again (owner decision, reverses an earlier
merge).** The merge existed only so one call could buy a veto over whether extraction ran at all
— that veto was removed when the word-count gate moved to ingest (``pipeline.py``'s module
docstring), and chunked extraction (below) is incompatible with a merged call anyway: several
per-chunk calls cannot produce one whole-transcript classification without disagreeing verdicts.
``triage.run_triage`` classifies the whole transcript once, on the cheap tier; its
``knowledge_types_present`` dominant type then steers :func:`run_chunked_extraction`, on the
strong tier — decide-then-act is preserved across two calls via that data dependency (extraction
cannot start building per-chunk prompts until it has the dominant type), the same ordering
guarantee the old two-call design always had, before either call being merged or split changed
anything about it.

**Extraction is chunked (owner decision, measured).** A single whole-transcript call skims a
long video; splitting it into ``distil/summary.py``'s ``chunk_transcript_text`` pieces and
extracting each separately found 2.7x more faithful items on a real ~8,600-word transcript, at
roughly proportional cost — see the PR that introduced :func:`run_chunked_extraction` for the
measurement. Two hazards this introduces, both handled here rather than assumed away:

- **Boundary damage** (an argument split across a chunk boundary becoming two half-formed
  items): each chunk after the first is prefixed with a small trailing overlap from the previous
  chunk, framed in the prompt as context the model may use to write one complete item spanning
  the boundary, but should not re-extract wholesale if it was already fully covered there.
- **Duplication** (the same point extracted twice — once per chunk, or restated across the
  overlap): ``_dedupe_near_duplicate_items`` folds items whose normalized statements are a close
  textual match (not just identical) before returning, reusing ``normalize.merge_duplicate_item``
  for the same field-folding ``normalize_items`` already does on exact duplicates — that exact-key
  pass still runs afterward in ``pipeline.py`` as a second, independent safety net.

``run_extraction`` (single whole-transcript call, no chunking) is unchanged and kept as a
standalone, independently-testable function — the gated eval suite and existing unit tests still
exercise it directly, isolated from chunking.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import get_args

from pydantic import ValidationError

from .faithfulness import _normalize
from .ingest import Transcript
from .llm import LLMClient
from .models import EntityKind, EntityMention, KnowledgeItem, KnowledgeType, Triage
from .normalize import merge_duplicate_item
from .prompts.extract import SYSTEM, build_extract_prompt
from .summary import chunk_transcript_text
from .triage import ParseError

logger = logging.getLogger(__name__)

_MAX_QUOTE_WORDS = 15
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_MAX_RETRIES = 2
_RETRY_SLEEP_SECONDS = 0.5

_VALID_KNOWLEDGE_TYPES = frozenset(get_args(KnowledgeType))
_VALID_ENTITY_KINDS = frozenset(get_args(EntityKind))

# A wholesale-garbage response (wrong shape, hallucinated fields, etc.) should still fail loudly
# rather than silently returning a near-empty result. Below half the items surviving validation
# is past what one or two isolated model mistakes would produce, so treat it as systemic and raise.
_MIN_SALVAGE_FRACTION = 0.5

# ~2,000-2,500 words per chunk — the exact size measured to find 2.7x more faithful items than a
# single whole-transcript call (see the module docstring). Deliberately the same default as
# summary.py's DEFAULT_CHUNK_CHARS: both are "a single completion covers this thoroughly without
# skimming" chunk sizes: not a coincidence, since that's exactly the size the measurement used.
DEFAULT_EXTRACT_CHUNK_CHARS = 12_000

# A modest trailing slice of the previous chunk, prepended as context so an argument split across
# a chunk boundary can still be written as one complete item. ~10% of the default chunk size:
# enough to carry the last sentence or two of setup, not enough to make the model re-extract a
# whole extra chunk's worth of already-covered material.
DEFAULT_EXTRACT_OVERLAP_CHARS = 1_200

# Two items are treated as the same point restated (not two genuinely different items) when their
# normalized statements are at least this similar. 0.85 is deliberately conservative — high enough
# that two items merely sharing topic/vocabulary don't collapse, but catches near-verbatim restates
# across a chunk boundary or an overlap section the model re-extracted despite the prompt's ask.
_NEAR_DUPLICATE_SIMILARITY = 0.85


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
class ChunkedExtractionResult:
    items: list[KnowledgeItem]
    chunk_count: int
    truncated: bool = False
    """True when *any* chunk's response had to be salvaged from an incomplete array (the
    output-token cap was hit, or the connection dropped mid-stream) rather than parsed as a
    complete response for that chunk — see ``_parse_items_json``. Rides through to
    ``pipeline.run_pipeline`` -> ``KBEntry.extraction_truncated`` exactly as the single-call path
    already does, so a partial chunk is never silently indistinguishable from a complete one."""


def run_chunked_extraction(
    transcript: Transcript,
    triage: Triage,
    client: LLMClient,
    *,
    chunk_chars: int | None = None,
    overlap_chars: int | None = None,
) -> ChunkedExtractionResult:
    """Extract knowledge items chunk-by-chunk instead of from the whole transcript in one call —
    see the module docstring's "Extraction is chunked" section for why and how the two hazards
    (boundary damage, duplication) are handled. ``triage`` supplies the dominant type extraction
    is conditioned on (decide-then-act, preserved across the two now-separate calls).
    """
    ktype = dominant_type(triage)
    target_chunk_chars = chunk_chars if chunk_chars is not None else _extract_chunk_chars_from_env()
    target_overlap_chars = (
        overlap_chars if overlap_chars is not None else _extract_overlap_chars_from_env()
    )
    chunks = chunk_transcript_text(transcript.full_text(), target_chunk_chars)
    if not chunks:
        return ChunkedExtractionResult(items=[], chunk_count=0, truncated=False)

    all_items: list[KnowledgeItem] = []
    any_truncated = False
    for i, chunk in enumerate(chunks):
        chunk_text = _with_overlap_context(chunk, chunks[i - 1] if i > 0 else None, target_overlap_chars)
        prompt = build_extract_prompt(ktype, chunk_text)
        data, truncated = _complete_with_retry(client, prompt, SYSTEM)
        items = _items_from_json(data, ktype)
        if len(chunks) > 1:
            # Each chunk's own _items_from_json call independently assigns k_01, k_02, ...
            # starting from 1 — only a genuine risk of collision (more than one chunk) pays the
            # cost of namespacing. A transcript short enough for one chunk (the common case,
            # and every existing single-call test fixture) keeps today's plain "k_01" ids
            # byte-for-byte, since nothing downstream needs the extra prefix.
            _renumber_item_ids(items, chunk_index=i)
        all_items.extend(items)
        any_truncated = any_truncated or truncated

    _truncate_overlong_quotes(all_items)
    _enforce_quote_discipline(all_items)
    deduped = _dedupe_near_duplicate_items(all_items)

    if any_truncated:
        logger.warning(
            "Chunked extraction: at least one of %d chunk(s) was truncated (output cap or "
            "dropped connection) and salvaged via partial recovery — that chunk's knowledge was "
            "NOT fully captured. %d item(s) survived overall.",
            len(chunks),
            len(deduped),
        )
    if len(deduped) != len(all_items):
        logger.info(
            "Chunked extraction: merged %d near-duplicate item(s) across %d chunk(s) (%d -> %d).",
            len(all_items) - len(deduped),
            len(chunks),
            len(all_items),
            len(deduped),
        )
    return ChunkedExtractionResult(items=deduped, chunk_count=len(chunks), truncated=any_truncated)


def _extract_chunk_chars_from_env() -> int:
    try:
        return int(os.environ.get("DISTIL_EXTRACT_CHUNK_CHARS", DEFAULT_EXTRACT_CHUNK_CHARS))
    except ValueError:
        return DEFAULT_EXTRACT_CHUNK_CHARS


def _extract_overlap_chars_from_env() -> int:
    try:
        return int(os.environ.get("DISTIL_EXTRACT_CHUNK_OVERLAP_CHARS", DEFAULT_EXTRACT_OVERLAP_CHARS))
    except ValueError:
        return DEFAULT_EXTRACT_OVERLAP_CHARS


def _with_overlap_context(chunk: str, previous_chunk: str | None, overlap_chars: int) -> str:
    """Prefix ``chunk`` with a trailing slice of ``previous_chunk`` (if any), framed as context
    the model may use to complete a boundary-spanning argument but should not wholesale
    re-extract. The first chunk gets no prefix — there is nothing before it to carry over."""
    if previous_chunk is None or overlap_chars <= 0:
        return chunk
    tail = previous_chunk[-overlap_chars:]
    # Snap to a sentence-ish boundary so the overlap doesn't open mid-word — good enough for
    # context (it is never itself the source of a new item's quote requirement).
    first_boundary = tail.find(". ")
    if 0 <= first_boundary < len(tail) - 2:
        tail = tail[first_boundary + 2 :]
    return (
        "[CONTEXT — already covered by a previous extraction pass. Do not re-extract items that "
        "are fully contained in this section. If an idea begun here continues into the NEW "
        "MATERIAL below, you may extract it as one complete item using text from either section "
        "for its verbatim quote.]\n"
        f"{tail}\n"
        "[END CONTEXT]\n\n"
        "[NEW MATERIAL]\n"
        f"{chunk}"
    )


def _renumber_item_ids(items: list[KnowledgeItem], *, chunk_index: int) -> None:
    """Namespace item ids by chunk so ids stay unique once chunks are combined — each chunk's
    own ``_items_from_json`` call independently assigns ``k_01``, ``k_02``, ... starting from 1."""
    for item in items:
        item.item_id = f"k_c{chunk_index:02d}_{item.item_id}"


def _dedupe_near_duplicate_items(items: list[KnowledgeItem]) -> list[KnowledgeItem]:
    """Fold items whose normalized statements are a close textual match (T-EC-dedup).

    Chunking's exact-duplicate case (the model re-extracts an item wholesale from an overlap
    section despite being asked not to) and its near-duplicate case (the same point independently
    restated in two chunks with slightly different wording) are both covered here by a fuzzy
    ratio rather than only an exact-string key — ``normalize_items``' own exact-key dedup still
    runs afterward in the pipeline as a second, independent safety net, but a paraphrase would
    slip past that one alone.
    """
    kept: list[KnowledgeItem] = []
    kept_keys: list[str] = []
    for item in items:
        key = _normalize(item.statement)
        match_index = None
        for i, existing_key in enumerate(kept_keys):
            if SequenceMatcher(None, key, existing_key).ratio() >= _NEAR_DUPLICATE_SIMILARITY:
                match_index = i
                break
        if match_index is not None:
            merge_duplicate_item(kept[match_index], item)
            continue
        kept.append(item)
        kept_keys.append(key)
    return kept


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
    makes a salvaged response distinguishable from a complete one (see
    ``ChunkedExtractionResult.truncated``) — a salvage that silently looks identical to success
    is exactly the defect this flag exists to prevent.
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
