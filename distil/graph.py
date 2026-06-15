"""Stage 5 — Graph linking. ARCHITECTURE.md §2; TESTING T-G1, T-G2.

Two steps. **Candidate lookup is deterministic** (a SQLite index query for entries sharing
topics/types — no LLM, T-G1). **Relation classification** asks the LLM to label each candidate
with one relation from the allowed enum; anything outside the enum (including ``none``) is
dropped rather than trusted (T-G2).

To respect the per-transcript LLM budget (≤ 4 calls total), only the top
``MAX_CANDIDATES`` candidates are classified.
"""

from __future__ import annotations

import json
import re

from .llm import LLMClient
from .models import KBEntry, RelatedEntry
from .prompts.graph import SYSTEM, build_relation_prompt
from .store import Store

MAX_CANDIDATES = 3
_ALLOWED = {"supports", "contradicts", "same_principle", "extends", "prerequisite_of"}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def link_graph(entry: KBEntry, store: Store, client: LLMClient) -> list[RelatedEntry]:
    # Topic overlap is the discriminating signal for candidacy; knowledge_type alone (e.g.
    # "everything is a heuristic") is too broad to link on, so we don't trigger on it.
    candidates = store.find_candidates(
        topics=entry.tags.topics,
        knowledge_types=[],
        exclude=entry.entry_id,
    )
    if not candidates:
        return []

    new_summary = _summarize(entry)
    edges: list[RelatedEntry] = []
    for cand in candidates[:MAX_CANDIDATES]:
        cand_entry = store.load_entry(cand.entry_id)
        prompt = build_relation_prompt(new_summary, cand.entry_id, _summarize(cand_entry))
        raw = client.complete(prompt, system=SYSTEM)
        relation = _parse_relation(raw)
        if relation in _ALLOWED:
            edges.append(RelatedEntry(target=cand.entry_id, relation=relation))
    return edges


def _summarize(entry: KBEntry) -> str:
    lines = [f"Title: {entry.source.title}"]
    for item in entry.knowledge_items[:5]:
        lines.append(f"- [{item.type}] {item.statement}")
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
