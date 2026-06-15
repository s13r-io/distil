"""Stage 3 — Normalize (PURE). ARCHITECTURE.md §2; TESTING T-N1..N4.

A deterministic gate after extraction. In order:

1. **Drop unverifiable items** — any item whose ``provenance.quote`` is not found in the
   transcript is removed (T-N2). This is the read-side-independent half of the no-fabrication
   guarantee and is never weakened.
2. **Backfill provenance** — locate the segment whose text contains the quote and copy its
   ``locator`` (and ``timestamp`` if it has one) onto the item, so untimestamped sources still
   get a stable pointer (T-N4).
3. **Merge near-duplicates** — items with the same normalized statement collapse to one (T-N1).

Stance is never altered: an opinion stays an opinion (T-N3).
"""

from __future__ import annotations

from .faithfulness import _normalize, quote_in_transcript
from .ingest import Transcript
from .models import KnowledgeItem


def normalize_items(items: list[KnowledgeItem], transcript: Transcript) -> list[KnowledgeItem]:
    verified: list[KnowledgeItem] = []
    seen_statements: dict[str, int] = {}

    for item in items:
        # (1) faithfulness gate — drop fabricated provenance.
        if not quote_in_transcript(item.provenance.quote, transcript):
            continue

        clone = item.model_copy(deep=True)
        _backfill_provenance(clone, transcript)

        # (3) near-duplicate merge keyed on the normalized statement.
        key = _normalize(clone.statement)
        if key in seen_statements:
            _merge_into(verified[seen_statements[key]], clone)
            continue
        seen_statements[key] = len(verified)
        verified.append(clone)

    return verified


def _backfill_provenance(item: KnowledgeItem, transcript: Transcript) -> None:
    needle = _normalize(item.provenance.quote)
    for seg in transcript.segments:
        if needle and needle in _normalize(seg.text):
            if item.provenance.locator is None:
                item.provenance.locator = seg.locator
            if item.provenance.timestamp is None and seg.timestamp is not None:
                item.provenance.timestamp = seg.timestamp
            return


def _merge_into(target: KnowledgeItem, dup: KnowledgeItem) -> None:
    """Fold a duplicate's incidental detail into the kept item without changing its meaning."""
    if target.rationale is None and dup.rationale is not None:
        target.rationale = dup.rationale
    if target.scope is None and dup.scope is not None:
        target.scope = dup.scope
    for pc in dup.preconditions:
        if pc not in target.preconditions:
            target.preconditions.append(pc)
    for g in dup.gotchas:
        if g not in target.gotchas:
            target.gotchas.append(g)
