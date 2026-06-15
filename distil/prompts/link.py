"""Application-link prompt (stage 4). TESTING T-L*.

Given knowledge items and the user's profile, propose concrete ways to apply each item to a
SPECIFIC goal or current-focus project. Every link must name a real ``linked_goal_id`` from
the profile. The code enforces goal validity and the novelty reservation after the model
responds; the prompt just asks for goal-tied, concrete scenarios.
"""

from __future__ import annotations

PROMPT_VERSION = "link/v1"

SYSTEM = (
    "You connect extracted knowledge to a specific user's goals. Each application link must "
    "target one real goal/focus id from the profile you are given and describe a concrete, "
    "personal scenario — not generic advice. Respond with a single JSON array and nothing else."
)

_TEMPLATE = """\
USER PROFILE (goals and focus you may link to):
{profile_goals}

KNOWLEDGE ITEMS:
{items}

Propose application links. Return EXACTLY a JSON array (no prose, no code fence):
[
  {{
    "knowledge_item_ids": ["k_01"],
    "linked_goal_id": "<one id from the profile goals above>",
    "application_form": "<checklist|trigger|flashcard|experiment|reference>",
    "scenario": "<concrete action tied to that goal>",
    "novelty_flag": false
  }}
]
Prefer the user's stated goals and active focus. Set novelty_flag=true only for an
intentionally orthogonal, serendipitous connection.
"""


def build_link_prompt(profile_goals: str, items: str) -> str:
    return _TEMPLATE.format(profile_goals=profile_goals, items=items)
