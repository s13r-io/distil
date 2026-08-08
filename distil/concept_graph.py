"""Concept↔concept typed edges (Phase 16, OKF Phase 3b design report §9 item 4).

The exact ``graph.py`` shape — deterministic candidate lookup, then a capped, enum-validated
LLM call per candidate — applied one more time at *concept* granularity instead of *entry*
granularity: candidacy is centroid-to-centroid cosine similarity
(``Store.find_concept_edge_candidates``, already capped/floored the same way
``find_concept_candidates`` is), and the LLM only labels each pair with one relation from
``{contrasts_with, builds_on, related}`` or drops it (anything outside the enum, including
``none``, is dropped rather than trusted — same discipline as ``graph._parse_relation``).

``link_concept_graph`` is pure DB-only (matches ``canonicalize_entry``'s shape); it runs once
per concept, right after that concept's claims are (re-)synthesized. ``run_concept_edges_stage``
is the one orchestration entry point that also talks to ``okf.py`` — mirrors
``run_canonicalize_stage``'s split between pure matching/synthesis functions and the one function
that renders. Idempotent: a concept's ``edges`` list is replaced wholesale on every recompute,
never appended to, and ``Store.prune_dangling_concept_edges`` drops any edge whose target concept
was deleted since the edge was last computed.
"""

from __future__ import annotations

import json
import re

from . import okf
from .llm import LLMClient
from .models import Concept, ConceptEdge
from .prompts.concept_graph import SYSTEM, build_concept_relation_prompt
from .store import Store

_ALLOWED = {"contrasts_with", "builds_on", "related"}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def link_concept_graph(concept: Concept, store: Store, client: LLMClient) -> list[ConceptEdge]:
    """Recompute ``concept``'s outgoing typed edges to other concepts.

    Deterministic candidate lookup (centroid cosine similarity) first; the LLM only labels each
    offered candidate pair. No candidates -> no LLM call (mirrors ``link_graph``'s T-G1 shortcut).
    """
    candidates = store.find_concept_edge_candidates(concept.concept_id)
    if not candidates:
        return []

    summary = _summarize(concept)
    edges: list[ConceptEdge] = []
    for cand in candidates:
        cand_concept = store.load_concept(cand.concept_id)
        if cand_concept is None:
            continue
        prompt = build_concept_relation_prompt(summary, cand.concept_id, _summarize(cand_concept))
        raw = client.complete(prompt, system=SYSTEM)
        relation = _parse_relation(raw)
        if relation in _ALLOWED:
            edges.append(ConceptEdge(target_concept_id=cand_concept.concept_id, relation=relation))
    return edges


def run_concept_edges_stage(touched: list[Concept], store: Store, client: LLMClient) -> list[Concept]:
    """Pipeline stage (design report §9 item 4, run after ``run_canonicalize_stage``): recompute
    typed edges for every concept actually synthesized this run — a ``touched`` concept still
    ``pending_synthesis`` (over the per-video synthesis cap) waits for a later catch-up pass
    alongside its eventual synthesis, the same capping discipline Phase 15.2 already applies.

    Also prunes edges left dangling by any concept deleted this run (``Store.
    prune_dangling_concept_edges``) and re-exports the OKF page for every concept whose edges
    changed — the touched ones, plus any survivor whose edges pointed at something now gone.
    """
    ready = [c for c in touched if not c.pending_synthesis]
    changed_ids: set[str] = set()
    for concept in ready:
        concept.edges = link_concept_graph(concept, store, client)
        store.save_concept(concept)
        changed_ids.add(concept.concept_id)

    changed_ids |= store.prune_dangling_concept_edges()

    changed: list[Concept] = []
    for concept_id in changed_ids:
        concept = store.load_concept(concept_id)
        if concept is not None:
            okf.export_concept(concept, store, store.okf_root)
            changed.append(concept)
    return changed


def _summarize(concept: Concept) -> str:
    lines = [f"Title: {concept.title}", f"Description: {concept.description}"]
    for claim in concept.claims[:3]:
        lines.append(f"- {claim.text}")
    return "\n".join(lines)


def _parse_relation(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
        return str(data.get("relation", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r'"relation"\s*:\s*"([a-z_]+)"', text)
        return match.group(1) if match else ""
