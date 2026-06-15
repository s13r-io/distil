"""Relation-classification prompt (stage 5). TESTING T-G2.

Candidate matching is a deterministic index lookup; the LLM is used only to label the
relationship between the new entry and each candidate, choosing from a fixed enum.
"""

from __future__ import annotations

PROMPT_VERSION = "graph/v1"

SYSTEM = (
    "You classify the relationship between a new knowledge entry and an existing one. Choose "
    "exactly one relation from the allowed set. If none fits, say 'none'. Respond with a single "
    "JSON object and nothing else."
)

_TEMPLATE = """\
NEW ENTRY:
{new_summary}

EXISTING CANDIDATE ENTRY ({candidate_id}):
{candidate_summary}

Pick the single best relation of the NEW entry TO the candidate:
- supports: the new entry agrees with / reinforces the candidate
- contradicts: the new entry disagrees with the candidate
- same_principle: both express the same underlying idea
- extends: the new entry builds on / adds to the candidate
- prerequisite_of: the candidate is a prerequisite for the new entry
- none: no meaningful relationship

Return EXACTLY: {{"relation": "<supports|contradicts|same_principle|extends|prerequisite_of|none>"}}
"""


def build_relation_prompt(new_summary: str, candidate_id: str, candidate_summary: str) -> str:
    return _TEMPLATE.format(
        new_summary=new_summary, candidate_id=candidate_id, candidate_summary=candidate_summary
    )
