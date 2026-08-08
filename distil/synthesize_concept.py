"""Concept-page synthesis (Phase 15.2, design report §4).

Reuses ``note.py``'s grounded-claim discipline (``_clean_note``/``_clean_grounded``/
``_fallback_note``) at concept granularity: the LLM may propose prose, but code enforces that
every claim traces to a real member of *this* concept before it is ever rendered. Malformed or
empty model output degrades to a deterministic one-liner built only from verified data — never
an exception, never an empty page.

Citation rendering (``render_claim``) is deliberately a separate, pure function: the model
never writes a citation string. The caller (``okf.export_concept``) resolves each validated
``item_id`` to an ``(okf_slug, timestamp)`` pair from verified member/entry data and passes that
resolved map in — code assembles the trailing parenthetical, never the model (report §4 step 4).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .llm import LLMClient
from .models import Concept, ConceptClaim, Entity, EntityClaim
from .prompts.synthesize_concept import (
    SYSTEM,
    SYSTEM_ENTITY,
    build_synthesize_entity_prompt,
    build_synthesize_prompt,
)

if TYPE_CHECKING:  # pragma: no cover - avoids a store<->okf<->synthesize_concept import cycle
    from .store import Store

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ConceptSynthesisError(ValueError):
    """Raised internally when a synthesis response cannot become usable claims."""


def synthesize_concept(concept: Concept, store: Store, client: LLMClient) -> Concept:
    """(Re-)synthesize ``concept.claims`` from its current ``members``.

    Never raises: malformed/empty model output falls back to a deterministic one-liner built
    from ``description`` and the bare member statements. Clears ``pending_synthesis`` and stamps
    ``body_model_version`` regardless of which path produced the claims.
    """
    member_statements = _load_member_statements(concept, store)
    valid_item_ids = {member.item_id for member in concept.members}

    claims: list[ConceptClaim] = []
    if concept.members:
        try:
            payload = _build_member_payload(concept, member_statements)
            raw = client.complete(
                build_synthesize_prompt(concept.title, concept.description, payload),
                system=SYSTEM,
            )
            claims = _clean_claims(_parse_claims_json(raw), valid_item_ids)
        except Exception:
            claims = []

    if not claims:
        claims = _fallback_claims(concept, member_statements)

    concept.claims = claims
    concept.body_model_version = _model_version(client)
    concept.pending_synthesis = False
    concept.updated_at = _now()
    return concept


def synthesize_entity(entity: Entity, store: Store, client: LLMClient) -> Entity:
    """(Re-)synthesize ``entity.claims`` from its current ``members`` — the exact
    ``synthesize_concept`` shape and fallback discipline, reusing the same private helpers
    (``_load_member_statements``/``_build_member_payload``/``_clean_claims``/etc.) since
    ``EntityMember`` carries the same ``entry_id``/``item_id`` identity ``ConceptMember`` does.
    Never raises: malformed/empty model output falls back to a deterministic one-liner."""
    member_statements = _load_member_statements(entity, store)
    valid_item_ids = {member.item_id for member in entity.members}

    claims: list[EntityClaim] = []
    if entity.members:
        try:
            payload = _build_member_payload(entity, member_statements)
            raw = client.complete(
                build_synthesize_entity_prompt(entity.title, entity.description, payload),
                system=SYSTEM_ENTITY,
            )
            claims = [
                EntityClaim(text=c.text, item_ids=c.item_ids)
                for c in _clean_claims(_parse_claims_json(raw), valid_item_ids)
            ]
        except Exception:
            claims = []

    if not claims:
        claims = [
            EntityClaim(text=c.text, item_ids=c.item_ids)
            for c in _fallback_claims(entity, member_statements)
        ]

    entity.claims = claims
    entity.body_model_version = _model_version(client)
    entity.pending_synthesis = False
    entity.updated_at = _now()
    return entity


def render_claim(
    claim: ConceptClaim | EntityClaim, citations: dict[str, tuple[str, str | None]]
) -> str:
    """Render one cleaned claim's prose with a citation appended by CODE, never by the model
    (design report §4 step 4). ``citations`` maps ``item_id`` -> ``(okf_slug, timestamp)``,
    resolved by the caller from verified member/entry data. Multiple ``item_ids`` render as
    multiple citations in one trailing parenthetical, comma-separated. Works identically for a
    ``ConceptClaim`` or an ``EntityClaim`` — both share the same ``{text, item_ids}`` shape."""
    parts: list[str] = []
    for item_id in claim.item_ids:
        resolved = citations.get(item_id)
        if resolved is None:
            continue
        slug, timestamp = resolved
        parts.append(f"{slug}, {timestamp}" if timestamp else slug)
    if not parts:
        return claim.text
    return f"{claim.text} ({', '.join(parts)})."


def _load_member_items(concept: Concept | Entity, store: Store) -> dict[str, Any]:
    """``item_id`` -> the real ``KnowledgeItem`` for every member whose owning entry still
    resolves. ``ConceptMember`` doesn't carry the full item (only ``quote``/``timestamp``,
    copied at match time), so it's looked up from the owning ``KBEntry`` via
    ``store.load_entry``. Shared by ``_load_member_statements`` (synthesis prompt input) and
    ``find_claim_contradictions`` (Phase 16 stance-conflict detection) — one entry-walk instead
    of two.
    """
    items: dict[str, Any] = {}
    entries_cache: dict[str, Any] = {}
    for member in concept.members:
        entry = entries_cache.get(member.entry_id, _MISSING)
        if entry is _MISSING:
            try:
                entry = store.load_entry(member.entry_id)
            except Exception:
                entry = None
            entries_cache[member.entry_id] = entry
        if entry is None:
            continue
        item = next((i for i in entry.knowledge_items if i.item_id == member.item_id), None)
        if item is not None:
            items[member.item_id] = item
    return items


def _load_member_statements(concept: Concept | Entity, store: Store) -> dict[str, str]:
    return {item_id: item.statement for item_id, item in _load_member_items(concept, store).items()}


def find_claim_contradictions(
    concept: Concept, store: Store
) -> dict[int, list[tuple[str, str, str]]]:
    """Deterministic contradiction detection (Phase 16, no LLM): for each claim (by index in
    ``concept.claims``), whether the ``KnowledgeItem.stance`` values of its cited members
    disagree. Pure comparison over already-verified item data — the same "derive, never invent"
    discipline :func:`render_claim` already applies to citations, just applied to detecting
    disagreement instead of assembling a citation string.

    Returns ``claim_index -> [(entry_id, item_id, stance), ...]`` — one row per cited member —
    only for claims where more than one distinct ``stance`` is actually present. Claims whose
    members all agree (or that cite only one member) are absent from the result.
    """
    items = _load_member_items(concept, store)
    members_by_item = {member.item_id: member for member in concept.members}
    conflicts: dict[int, list[tuple[str, str, str]]] = {}
    for idx, claim in enumerate(concept.claims):
        rows = [
            (members_by_item[item_id].entry_id, item_id, items[item_id].stance)
            for item_id in claim.item_ids
            if item_id in items and item_id in members_by_item
        ]
        if len({stance for _entry_id, _item_id, stance in rows}) > 1:
            conflicts[idx] = rows
    return conflicts


_MISSING = object()


def _build_member_payload(concept: Concept | Entity, statements: dict[str, str]) -> list[dict]:
    return [
        {
            "item_id": member.item_id,
            "entry_id": member.entry_id,
            "statement": statements.get(member.item_id, ""),
            "quote": member.quote,
            "timestamp": member.timestamp,
        }
        for member in concept.members
    ]


def _parse_claims_json(raw: str) -> list[Any]:
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ConceptSynthesisError(f"Synthesis response was not JSON: {raw[:120]!r}") from exc
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            raise ConceptSynthesisError(f"Synthesis response was not JSON: {raw[:120]!r}") from exc2
    if not isinstance(data, list):
        raise ConceptSynthesisError("Synthesis response must be a JSON array.")
    return data


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped


def _clean_claims(raw_claims: list[Any], valid_item_ids: set[str]) -> list[ConceptClaim]:
    """Drop any claim whose ``item_ids`` don't ALL resolve to real members of this concept, or
    whose text/item_ids are empty (design report §4 step 3) — a mixed valid/invalid citation is
    not trusted at concept granularity, unlike ``note.py``'s per-id filtering."""
    cleaned: list[ConceptClaim] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        text = str(raw_claim.get("text", "")).strip()
        raw_ids = raw_claim.get("item_ids", [])
        if not text or not isinstance(raw_ids, list) or not raw_ids:
            continue
        item_ids = [str(i) for i in raw_ids]
        if not all(item_id in valid_item_ids for item_id in item_ids):
            continue
        deduped = list(dict.fromkeys(item_ids))
        cleaned.append(ConceptClaim(text=text, item_ids=deduped))
    return cleaned


def _fallback_claims(concept: Concept | Entity, statements: dict[str, str]) -> list[ConceptClaim]:
    item_ids = [member.item_id for member in concept.members if member.item_id in statements]
    parts = [concept.description.strip()] if concept.description.strip() else []
    parts.extend(statements[item_id].strip() for item_id in item_ids if statements[item_id].strip())
    text = " ".join(part for part in parts if part)
    if not text or not item_ids:
        return []
    return [ConceptClaim(text=text, item_ids=item_ids)]


def _model_version(client: LLMClient) -> str:
    return getattr(client, "model", "") or os.environ.get("DISTIL_MODEL", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
