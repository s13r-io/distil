"""Triage prompt (stage 1). Versioned string + builder.

The model classifies the knowledge types present, density, transcript-loss level (with
verbatim evidence), and an overall verdict. It must be honest: low-value input should yield
``little_to_extract`` rather than manufactured insight (PRD FR12).
"""

from __future__ import annotations

PROMPT_VERSION = "triage/v1"

SYSTEM = (
    "You are the triage stage of a knowledge-distillation pipeline. You classify a transcript; "
    "you do NOT extract or summarize it. Be honest: if a transcript carries little extractable "
    "knowledge (entertainment, chit-chat, filler), say so with the verdict 'little_to_extract' "
    "rather than inventing insights. Respond with a single JSON object and nothing else."
)

_TEMPLATE = """\
Classify the transcript below.

Return EXACTLY this JSON shape (no prose, no code fence):
{{
  "knowledge_types_present": [{{"type": "<heuristic|procedural|declarative|conceptual|experiential|opinion>", "share": <0..1>}}],
  "density": "<low|medium|high>",
  "transcript_loss": {{"level": "<low|medium|high>", "evidence": ["<verbatim phrase showing lost visual/context>"]}},
  "verdict": "<rich|mixed|little_to_extract>"
}}

Rules:
- "share" values should sum to roughly 1.0 across the types present.
- transcript_loss is HIGH when the speaker leans on visuals the transcript can't capture
  ("as you can see here", "this line", "look at that"). Put those exact phrases in "evidence".
- verdict "little_to_extract" when there is almost no reusable knowledge.

TRANSCRIPT:
{transcript}
"""


def build_triage_prompt(transcript_text: str) -> str:
    return _TEMPLATE.format(transcript=transcript_text)
