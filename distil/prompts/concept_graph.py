"""Concept-to-concept relation-classification prompt (Phase 16, design report §9 item 4).

Candidate lookup is deterministic (centroid-to-centroid cosine similarity,
``Store.find_concept_edge_candidates``); the LLM only labels the relationship between two
concepts, choosing from a fixed enum — the exact ``prompts/graph.py`` pattern, applied one more
time between concepts instead of entries.
"""

from __future__ import annotations

PROMPT_VERSION = "concept_graph/v1"

SYSTEM = (
    "You classify the relationship between two knowledge concepts. Choose exactly one relation "
    "from the allowed set. If none fits, say 'none'. Respond with a single JSON object and "
    "nothing else."
)

_TEMPLATE = """\
CONCEPT:
{concept_summary}

CANDIDATE CONCEPT ({candidate_id}):
{candidate_summary}

Pick the single best relation of CONCEPT to the candidate:
- contrasts_with: the two concepts are competing or opposing approaches to the same problem
- builds_on: the concept builds on / extends the candidate's idea
- related: the concepts are meaningfully connected but neither contrasts with nor builds on it
- none: no meaningful relationship

Return EXACTLY: {{"relation": "<contrasts_with|builds_on|related|none>"}}
"""


def build_concept_relation_prompt(
    concept_summary: str, candidate_id: str, candidate_summary: str
) -> str:
    return _TEMPLATE.format(
        concept_summary=concept_summary,
        candidate_id=candidate_id,
        candidate_summary=candidate_summary,
    )
