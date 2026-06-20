"""Reader-facing note synthesis prompt (Note v1).

Atomic extraction remains the evidence layer. This prompt asks the model to turn verified
items into a useful teaching note while citing item IDs for every claim.
"""

from __future__ import annotations

import json

from distil.models import ApplicationLink, KnowledgeItem, Triage

PROMPT_VERSION = "note/v1"

SYSTEM = (
    "You turn verified transcript-derived knowledge items into a coherent teaching note. "
    "Use ONLY the provided items and application links. Every claim must cite one or more "
    "provided item_ids. Preserve stance: do not present opinions or personal experiences as "
    "facts. Prefer useful explanation over exhaustiveness. Respond with a single JSON object "
    "and nothing else."
)

_TEMPLATE = """\
SOURCE TITLE:
{source_title}

TRIAGE:
{triage}

VERIFIED KNOWLEDGE ITEMS:
{items}

APPLICATION LINKS:
{links}

Return EXACTLY this JSON object (no prose, no code fence):
{{
  "title": "<short useful note title>",
  "core_takeaway": {{"text": "<the main lesson>", "item_ids": ["k_01"]}},
  "key_points": [{{"text": "<important point>", "item_ids": ["k_01"]}}],
  "why_it_matters": [{{"text": "<why this is useful>", "item_ids": ["k_01"]}}],
  "how_to_apply": [
    {{"text": "<concrete application>", "item_ids": ["k_01"], "application_link_ids": ["a_01"]}}
  ],
  "caveats": [{{"text": "<scope, limitation, or risk>", "item_ids": ["k_01"]}}],
  "review_questions": [{{"question": "<question that helps retain the idea>", "item_ids": ["k_01"]}}],
  "topics": ["<short topic>", "<another topic>"]
}}

Rules:
- Use only the verified knowledge items and application links above.
- Every explanatory claim must cite one or more provided item_ids.
- Preserve stance: opinion stays opinion, personal experience stays personal experience.
- Prefer teaching value over listing every item.
- Do not invent topics unsupported by the items.
"""


def build_note_prompt(
    source_title: str,
    triage: Triage,
    items: list[KnowledgeItem],
    links: list[ApplicationLink],
) -> str:
    item_payload = [
        {
            "item_id": item.item_id,
            "type": item.type,
            "stance": item.stance,
            "speaker_confidence": item.speaker_confidence,
            "statement": item.statement,
            "rationale": item.rationale,
            "scope": item.scope,
            "order_index": item.order_index,
            "preconditions": item.preconditions,
            "gotchas": item.gotchas,
            "provenance_quote": item.provenance.quote,
            "provenance_timestamp": item.provenance.timestamp,
            "provenance_locator": item.provenance.locator,
        }
        for item in items
    ]
    link_payload = [
        {
            "link_id": link.link_id,
            "knowledge_item_ids": link.knowledge_item_ids,
            "linked_goal_id": link.linked_goal_id,
            "application_form": link.application_form,
            "scenario": link.scenario,
            "novelty_flag": link.novelty_flag,
        }
        for link in links
    ]
    triage_payload = {
        "knowledge_types_present": [
            {"type": share.type, "share": share.share}
            for share in triage.knowledge_types_present
        ],
        "density": triage.density,
        "transcript_loss": {
            "level": triage.transcript_loss.level,
            "evidence": triage.transcript_loss.evidence,
        },
        "verdict": triage.verdict,
    }
    return _TEMPLATE.format(
        source_title=source_title,
        triage=json.dumps(triage_payload, ensure_ascii=False, indent=2),
        items=json.dumps(item_payload, ensure_ascii=False, indent=2),
        links=json.dumps(link_payload, ensure_ascii=False, indent=2),
    )
