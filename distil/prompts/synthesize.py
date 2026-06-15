"""Grounded-synthesis prompt (read layer). ARCHITECTURE.md §9; TESTING T-Q3, T-Q6.

The model answers a question using ONLY the retrieved notes it is given. Every claim must
trace to a provided note id; it must not use outside knowledge. When notes conflict, it must
surface the disagreement rather than silently choosing one.
"""

from __future__ import annotations

PROMPT_VERSION = "synthesize/v1"

SYSTEM = (
    "You answer using ONLY the numbered notes provided. Do NOT use any outside knowledge. "
    "Every sentence in your answer must be grounded in at least one note, cited as [item_id]. "
    "If the notes conflict, say so explicitly and present both sides — never silently pick one. "
    "If the notes do not actually answer the question, say you don't have notes on it. "
    "Respond with a single JSON object and nothing else."
)

_TEMPLATE = """\
QUESTION: {question}

NOTES (the ONLY information you may use):
{notes}

Return EXACTLY this JSON (no prose, no code fence):
{{
  "answer": "<grounded answer; cite each claim as [item_id]>",
  "cited_item_ids": ["<every item_id you used>"],
  "conflict": "<describe any disagreement between notes, or null>"
}}
"""


def build_synthesis_prompt(question: str, notes_block: str) -> str:
    return _TEMPLATE.format(question=question, notes=notes_block)
