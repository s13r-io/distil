"""Stage 8 — canonicalize matching engine (Phase 15.1, OKF Phase 3a design report §1, §3, §5).

Same two-step shape ``graph.py`` already established — deterministic candidates, then a
capped, enum-validated LLM call — applied at *item* granularity instead of *entry* granularity,
with embedding similarity (``Store.find_concept_candidates``) as the primary candidacy signal
instead of topic overlap (too coarse within a single video — items in the same transcript can
belong to different concepts despite sharing one ``entry.tags.topics`` set).

This module produces DB rows only: no OKF export, no synthesis, no pipeline wiring. Those land
in Phase 15.2/15.3. The invariant this module guarantees is **idempotent membership**: re-filing
the same entry retracts its prior memberships before recomputing them, so it reproduces the same
result rather than accumulating duplicates.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .llm import LLMClient
from .models import Concept, ConceptMember, KBEntry, KnowledgeItem
from .okf import slugify
from .prompts.canonicalize import SYSTEM, build_canonicalize_prompt
from .store import ConceptCandidate, Store

_ALLOWED_DECISIONS = {"match", "new", "reject"}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TITLE_NORM = re.compile(r"[^a-z0-9]+")


def canonicalize_entry(entry: KBEntry, store: Store, client: LLMClient) -> list[Concept]:
    """Decide match/new/reject for every item in ``entry`` and materialize the result.

    Idempotent (design report §5): this entry's existing concept memberships are retracted
    first, so re-canonicalizing the same entry reproduces the same membership rather than
    duplicating it. Returns every concept touched (matched into, or newly created) this run.
    """
    store.retract_entry_concept_memberships(entry.entry_id)

    if not entry.knowledge_items:
        return []

    item_vectors = {
        item_id: vec for item_id, e_id, vec in store.iter_item_vectors() if e_id == entry.entry_id
    }
    candidates_by_item: dict[str, list[ConceptCandidate]] = {}
    for item in entry.knowledge_items:
        vec = item_vectors.get(item.item_id)
        candidates_by_item[item.item_id] = (
            store.find_concept_candidates(vec, entry.tags.topics, item.statement)
            if vec is not None
            else []
        )

    raw = client.complete(
        build_canonicalize_prompt(entry.source.title, _build_payloads(entry, candidates_by_item)),
        system=SYSTEM,
    )
    items_by_id = {item.item_id: item for item in entry.knowledge_items}
    decisions = _parse_decisions(raw, items_by_id.keys())

    touched: dict[str, Concept] = {}
    new_groups: dict[str, dict] = {}
    for item_id, decision in decisions.items():
        item = items_by_id[item_id]
        kind = decision.get("decision")
        if kind == "match":
            _apply_match(entry, item, decision, candidates_by_item[item_id], store, touched)
        elif kind == "new":
            _collect_new(item, decision, new_groups)
        # "reject" (or anything defaulted to it) produces no record at all.

    for group in new_groups.values():
        concept = _materialize_new_concept(entry, group, store)
        touched[concept.concept_id] = concept

    for concept in touched.values():
        store.save_concept(concept)
    return list(touched.values())


def _build_payloads(
    entry: KBEntry, candidates_by_item: dict[str, list[ConceptCandidate]]
) -> list[dict]:
    return [
        {
            "item_id": item.item_id,
            "statement": item.statement,
            "quote": item.provenance.quote,
            "candidates": [
                {"concept_id": c.concept_id, "title": c.title, "description": c.description}
                for c in candidates_by_item[item.item_id]
            ],
        }
        for item in entry.knowledge_items
    ]


def _apply_match(
    entry: KBEntry,
    item: KnowledgeItem,
    decision: dict,
    candidates: list[ConceptCandidate],
    store: Store,
    touched: dict[str, Concept],
) -> None:
    concept_id = str(decision.get("concept_id", ""))
    offered = {c.concept_id for c in candidates}
    if concept_id not in offered:
        return  # untrusted concept_id: no honest signal either way, so treat as reject
    concept = touched.get(concept_id)
    if concept is None:
        concept = store.load_concept(concept_id)
        if concept is None:
            return
    concept.members.append(_member(entry, item))
    concept.updated_at = _now()
    touched[concept_id] = concept


def _collect_new(item: KnowledgeItem, decision: dict, new_groups: dict[str, dict]) -> None:
    title = str(decision.get("title", "")).strip()
    if not title:
        return  # nothing to slug
    key = _normalize_title(title)
    group = new_groups.setdefault(
        key,
        {"title": title, "description": str(decision.get("description", "")).strip(), "items": []},
    )
    group["items"].append(item)


def _materialize_new_concept(entry: KBEntry, group: dict, store: Store) -> Concept:
    concept_id = _slug_for_concept(group["title"], store)
    now = _now()
    return Concept(
        concept_id=concept_id,
        title=group["title"],
        description=group["description"],
        members=[_member(entry, item) for item in group["items"]],
        created_at=now,
        updated_at=now,
    )


def _member(entry: KBEntry, item: KnowledgeItem) -> ConceptMember:
    return ConceptMember(
        entry_id=entry.entry_id,
        item_id=item.item_id,
        quote=item.provenance.quote,
        timestamp=item.provenance.timestamp,
    )


def _slug_for_concept(title: str, store: Store) -> str:
    base = slugify(title) or "concept"
    if store.load_concept(base) is None:
        return base
    suffix = 2
    while store.load_concept(f"{base}-{suffix}") is not None:
        suffix += 1
    return f"{base}-{suffix}"


def _normalize_title(title: str) -> str:
    return _TITLE_NORM.sub(" ", title.lower()).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_decisions(raw: str, item_ids) -> dict[str, dict]:
    """Defensively-fenced JSON-array parse, mirroring ``graph._parse_relation`` /
    ``note._parse_object``: strip code fences, regex-recover an array from prose, and never
    raise on malformed output. Decisions for unknown/duplicate ``item_id``s are dropped."""
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            data = []
    if not isinstance(data, list):
        data = []

    valid_ids = set(item_ids)
    out: dict[str, dict] = {}
    for raw_decision in data:
        if not isinstance(raw_decision, dict):
            continue
        item_id = raw_decision.get("item_id")
        if item_id not in valid_ids or item_id in out:
            continue
        decision = raw_decision.get("decision")
        if decision not in _ALLOWED_DECISIONS:
            decision = "reject"
        out[item_id] = {**raw_decision, "decision": decision}
    return out
