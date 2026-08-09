"""Merged triage+extraction prompt (stage 1+2, one strong-tier call). Versioned string + builder.

Owner decision (supersedes the old separate triage.py/extract.py calls): triage's short-circuit
is gone (see pipeline.py's module docstring), so the only reason for a cheap classification
pass *before* extraction — buying a veto over whether extraction happened at all — no longer
exists. The strong model now reads the full transcript exactly once. It states its
classification (dominant knowledge type, density, transcript-loss) FIRST, in a ``<TRIAGE>``
section, then extracts knowledge items SECOND, in an ``<ITEMS>`` section, conditioned on what
it just classified — preserving the decide-then-act sequencing the old two-call design bought
via call ordering, but within one response instead of two.

Two tagged sections (rather than one merged JSON object) is a deliberate parsing choice: it lets
``distil/extract.py`` reuse both existing, separately-tested parsers unchanged — triage's small
JSON-object parser for ``<TRIAGE>``, and extraction's JSON-array parser (including its
truncated-response recovery for a long items array cut off by the output-token cap) for
``<ITEMS>`` — rather than inventing truncation recovery for a value nested inside one bigger
object.

Type tailoring collapses to one merged shape: of the six knowledge types, only ``heuristic``
(rationale/scope) and ``procedural`` (order_index/preconditions/gotchas) add fields at all, so
every item shape includes both, generically, rather than routing to a per-type template chosen
ahead of time (impossible here — the type isn't known until the model states it). ``type`` is
decided per item, not fixed for the whole array, exactly as the old design already tolerated via
``extract.py``'s ``_repair_type`` (a different-but-valid type on one item was always allowed).
"""

from __future__ import annotations

PROMPT_VERSION = "triage_extract/v1"

SYSTEM = (
    "You read a transcript ONCE and do two things, in order, in a single response. FIRST, "
    "classify it: identify the knowledge types present, how dense it is, and how much is lost "
    "without the visuals. Density and transcript-loss are purely informational — they never "
    "change what you extract. Be honest: if a transcript carries little extractable knowledge "
    "(entertainment, chit-chat, filler), say so with the verdict 'little_to_extract' rather "
    "than inventing insights. SECOND, extract atomic, self-contained knowledge items — "
    "primarily of the dominant type you just identified, though a clearly-present item of a "
    "different type may also appear — each rewritten in clear neutral language. NEVER invent "
    "content: every item must be supported by a SHORT verbatim quote (fewer than 15 words) "
    "copied exactly from the transcript. There are THREE SEPARATE, NEVER-mixed vocabularies: "
    "`type` (heuristic/procedural/declarative/conceptual/experiential/opinion) classifies the "
    "kind of knowledge, decided per item. `stance` (fact/opinion/personal_experience) "
    "separately preserves whether the speaker was stating a fact, an opinion, or a personal "
    "story — mark opinions as opinion, personal stories as personal_experience, never dress an "
    "opinion up as fact. `personal_experience` is a `stance`, never a `type`. Third and "
    "separate again: each item may list `entities` — named tools, people, or organizations "
    "mentioned in that same sentence, each with its own `kind` field (tool/person/"
    "organization). `kind` is NEVER a `type` and NEVER a `stance`, and `type`/`stance` are "
    "NEVER a `kind`. Only extract entities that are genuinely named — do not invent an entity "
    "for a generic noun. Respond with EXACTLY the two-section format you are given and nothing "
    "else — no prose, no code fences, outside the tags."
)

_TEMPLATE = """\
Read the transcript below once. Return your response in EXACTLY this format: a <TRIAGE> \
section, then an <ITEMS> section, and nothing outside them.

<TRIAGE>
{{"knowledge_types_present": [{{"type": "<heuristic|procedural|declarative|conceptual|experiential|opinion>", "share": <0..1>}}], "density": "<low|medium|high>", "transcript_loss": {{"level": "<low|medium|high>", "evidence": ["<verbatim phrase showing lost visual/context>"]}}, "verdict": "<rich|mixed|little_to_extract>"}}
</TRIAGE>
<ITEMS>
[
  {{
    "type": "<heuristic|procedural|declarative|conceptual|experiential|opinion>",
    "statement": "<the knowledge, rewritten in your own words>",
    "stance": "<fact|opinion|personal_experience>",
    "speaker_confidence": "<low|medium|high>",
    "rationale": "<why it works, or null - heuristic items only>",
    "scope": "<when it applies/doesn't, or null - heuristic items only>",
    "order_index": <0-based step number, or null - procedural items only>,
    "preconditions": [],
    "gotchas": [],
    "provenance": {{"quote": "<<15-word verbatim quote from the transcript>", "timestamp": null, "locator": null}},
    "entities": [
      {{"name": "<canonical name of the tool/person/organization>", "kind": "<tool|person|organization>", "description": "<one short clause, in your own words>", "quote": "<<15-word verbatim quote from the transcript>", "timestamp": null}}
    ]
  }}
]
</ITEMS>

Rules:
- TRIAGE "share" values should sum to roughly 1.0 across the types present.
- transcript_loss is HIGH when the speaker leans on visuals the transcript can't capture
  ("as you can see here", "this line", "look at that"). Put those exact phrases in "evidence".
- verdict "little_to_extract" when there is almost no reusable knowledge — informational only,
  it does not stop or change extraction below.
- ITEMS: `type` is decided per item, not fixed for the whole array — extract primarily the
  dominant type you named in TRIAGE, but include any other clearly-present type too.
- "rationale"/"scope" apply only to heuristic items; "order_index"/"preconditions"/"gotchas"
  apply only to procedural items; omit or null them for every other type.
- `entities` is a separate, optional list per item — omit it or leave it `[]` when this item
  names no tool, person, or organization.
- Only include items genuinely supported by the transcript. If there is nothing to extract,
  return an empty items array.

TRANSCRIPT:
{transcript}
"""


def build_triage_extract_prompt(transcript_text: str) -> str:
    return _TEMPLATE.format(transcript=transcript_text)
