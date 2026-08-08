"""Canonicalize match/new/reject prompt (Stage 8, Phase 15.1).

Candidate gathering is deterministic (embedding similarity vs concept centroids, unioned with
a token-overlap backstop — see ``store.find_concept_candidates``). The LLM is used only to
decide, per item and using only the offered candidates, whether it matches an existing concept,
starts a new one, or isn't concept-worthy. One batched call covers the whole video.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "canonicalize/v1"

SYSTEM = (
    "You decide whether each new knowledge item is the same idea as an existing concept, a "
    "brand-new concept, or not concept-worthy. Use ONLY the candidates provided for each item — "
    "never invent a concept_id that isn't listed. Respond with a single JSON array and nothing "
    "else."
)

_TEMPLATE = """\
VIDEO: {source_title}

ITEMS AND THEIR CANDIDATE CONCEPTS:
{items_block}

For each item, return exactly one decision:
- {{"item_id": "...", "decision": "match", "concept_id": "<one of its listed candidates>"}}
- {{"item_id": "...", "decision": "new", "title": "...", "description": "<one sentence>"}}
- {{"item_id": "...", "decision": "reject"}}

Rules:
- "match" only to a concept_id that was listed as a candidate for that item.
- "new" only when no candidate is really the same idea (same claim, not just the same topic).
- "reject" for items that are personal anecdotes or facts about the speaker, not durable ideas.
Return a JSON array with exactly one decision object per item_id, in the order given.
"""


def build_canonicalize_prompt(source_title: str, item_payloads: list[dict]) -> str:
    """``item_payloads``: one dict per item with ``item_id``, ``statement``, ``quote``, and a
    ``candidates`` list of ``{concept_id, title, description}`` (empty when the deterministic
    pool found nothing — the item can still be proposed as ``new`` or ``reject``)."""
    return _TEMPLATE.format(
        source_title=source_title,
        items_block=json.dumps(item_payloads, ensure_ascii=False, indent=2),
    )


# ---- Entity match/new/reject (Phase D — reuses this exact shape one granularity down) -----

SYSTEM_ENTITIES = (
    "You decide whether each mentioned entity (a tool, person, or organization) is the same "
    "real-world thing as an existing entity, a brand-new entity, or not worth keeping. Use ONLY "
    "the candidates provided for each mention — never invent an entity_id that isn't listed. "
    "The entity's kind (tool/person/organization) is already fixed and is not yours to decide. "
    "Respond with a single JSON array and nothing else."
)

_ENTITY_TEMPLATE = """\
VIDEO: {source_title}

MENTIONS AND THEIR CANDIDATE ENTITIES:
{mentions_block}

For each mention, return exactly one decision:
- {{"mention_key": "...", "decision": "match", "entity_id": "<one of its listed candidates>"}}
- {{"mention_key": "...", "decision": "new", "title": "<canonical display name>", "description": "<one sentence>"}}
- {{"mention_key": "...", "decision": "reject"}}

Rules:
- "match" only to an entity_id that was listed as a candidate for that mention.
- "new" only when no candidate is really the same tool/person/organization.
- "reject" for a mention that's too vague or generic to be worth keeping as its own page.
Return a JSON array with exactly one decision object per mention_key, in the order given.
"""


def build_entity_canonicalize_prompt(source_title: str, mention_payloads: list[dict]) -> str:
    """``mention_payloads``: one dict per entity mention with ``mention_key``, ``name``,
    ``kind``, ``description``, ``quote``, and a ``candidates`` list of
    ``{entity_id, title, description}`` (already pre-filtered to the same ``kind``)."""
    return _ENTITY_TEMPLATE.format(
        source_title=source_title,
        mentions_block=json.dumps(mention_payloads, ensure_ascii=False, indent=2),
    )
