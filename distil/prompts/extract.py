"""Extraction prompts (stage 2), routed by knowledge type. TESTING T-E*.

Each type gets a tailored instruction so the output shape fits the content (procedural keeps
order; heuristic keeps rationale + scope). Every item must carry a SHORT verbatim quote from
the transcript — the faithfulness anchor. The model rewrites statements in our words but the
``provenance.quote`` must be copied verbatim and under 15 words.
"""

from __future__ import annotations

PROMPT_VERSION = "extract/v1"

SYSTEM = (
    "You are the extraction stage of a knowledge-distillation pipeline. Extract atomic, "
    "self-contained knowledge items, each rewritten in clear neutral language. NEVER invent "
    "content: every item must be supported by a SHORT verbatim quote (fewer than 15 words) "
    "copied exactly from the transcript. Preserve stance — mark opinions as opinion, personal "
    "stories as personal_experience, never dress an opinion up as fact. Respond with a single "
    "JSON array and nothing else."
)

_COMMON_SHAPE = """\
Return EXACTLY a JSON array of items with this shape (no prose, no code fence):
[
  {{
    "type": "{type}",
    "statement": "<the knowledge, rewritten in your own words>",
    "stance": "<fact|opinion|personal_experience>",
    "speaker_confidence": "<low|medium|high>",
    {type_fields}
    "provenance": {{"quote": "<<15-word verbatim quote from the transcript>", "timestamp": null, "locator": null}}
  }}
]
Only include items that are genuinely supported by the transcript. If there is nothing of this
type, return an empty array []."""

_TYPE_FIELDS = {
    "heuristic": '"rationale": "<why it works, or null>", "scope": "<when it applies/doesn\'t, or null>",',
    "procedural": '"order_index": <0-based step number>, "preconditions": [], "gotchas": [],',
    "declarative": "",
    "conceptual": "",
    "experiential": "",
    "opinion": "",
}

_TEMPLATE = """\
Extract the {type} knowledge from the transcript below.

{shape}

TRANSCRIPT:
{transcript}
"""


def build_extract_prompt(knowledge_type: str, transcript_text: str) -> str:
    fields = _TYPE_FIELDS.get(knowledge_type, "")
    shape = _COMMON_SHAPE.format(type=knowledge_type, type_fields=fields)
    return _TEMPLATE.format(type=knowledge_type, shape=shape, transcript=transcript_text)
