"""Concept-page synthesis prompt (Phase 15.2, design report §4).

The model only ever writes ``{text, item_ids}`` claim objects — never a citation string.
Citation rendering is done deterministically by code from verified member data
(``distil/synthesize_concept.py:render_claim``), the one deliberate faithfulness-driven
departure from the reference OKF bundle's model-authored inline citations (report §4 step 4).
"""

from __future__ import annotations

import json

PROMPT_VERSION = "synthesize_concept/v1"

SYSTEM = (
    "You write a grounded synthesis of one concept across multiple videos. Use ONLY the "
    "provided items. Every claim must cite one or more item_ids from the list. If two members "
    "disagree, write both and cite both — do not silently pick one. Never include a citation, "
    "video name, or timestamp in the claim text itself; cite only via item_ids. Respond with a "
    "single JSON array and nothing else."
)

_TEMPLATE = """\
CONCEPT: {title}
DESCRIPTION: {description}

MEMBERS (one knowledge item per video that discusses this concept):
{members_block}

Return a JSON array of claim objects, each exactly:
{{"text": "<synthesized sentence or short paragraph>", "item_ids": ["k_01", "k_02"]}}

Rules:
- Use only the members listed above.
- Every claim must cite one or more of the listed item_ids.
- If two members disagree, write both and cite both — never silently pick one.
- Do not write citations, video names, or timestamps into the text; cite only via item_ids.
Return a JSON array and nothing else.
"""


def build_synthesize_prompt(title: str, description: str, members: list[dict]) -> str:
    """``members``: one dict per concept member with ``item_id``, ``entry_id``, ``statement``,
    ``quote``, and ``timestamp`` (design report §4 step 2)."""
    return _TEMPLATE.format(
        title=title,
        description=description,
        members_block=json.dumps(members, ensure_ascii=False, indent=2),
    )


# ---- Entity synthesis (Phase D — same claim shape/discipline, one level down) ------------

SYSTEM_ENTITY = (
    "You write a grounded synthesis of one entity (a tool, person, or organization) across "
    "multiple videos. Use ONLY the provided items. Every claim must cite one or more item_ids "
    "from the list. If two members disagree, write both and cite both — do not silently pick "
    "one. Never include a citation, video name, or timestamp in the claim text itself; cite "
    "only via item_ids. Respond with a single JSON array and nothing else."
)

_ENTITY_TEMPLATE = """\
ENTITY: {title}
DESCRIPTION: {description}

MENTIONS (one knowledge item per video that mentions this entity):
{members_block}

Return a JSON array of claim objects, each exactly:
{{"text": "<synthesized sentence or short paragraph>", "item_ids": ["k_01", "k_02"]}}

Rules:
- Use only the members listed above.
- Every claim must cite one or more of the listed item_ids.
- If two members disagree, write both and cite both — never silently pick one.
- Do not write citations, video names, or timestamps into the text; cite only via item_ids.
Return a JSON array and nothing else.
"""


def build_synthesize_entity_prompt(title: str, description: str, members: list[dict]) -> str:
    """``members``: one dict per entity member with ``item_id``, ``entry_id``, ``statement``,
    ``quote``, and ``timestamp`` — the exact ``build_synthesize_prompt`` shape."""
    return _ENTITY_TEMPLATE.format(
        title=title,
        description=description,
        members_block=json.dumps(members, ensure_ascii=False, indent=2),
    )
