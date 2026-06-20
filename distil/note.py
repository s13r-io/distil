"""Note v1 — turn verified items into a reader-facing teaching note.

The LLM may propose prose, but the code keeps the faithfulness boundary: every note section
must cite valid extracted item IDs. Invalid or malformed output degrades to a deterministic
fallback built only from verified items and application links.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .llm import LLMClient
from .models import (
    ActionStep,
    ApplicationLink,
    DistilledNote,
    GroundedText,
    KnowledgeItem,
    ReviewQuestion,
    Triage,
)
from .prompts.note import SYSTEM, build_note_prompt

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TOPIC_CHARS = re.compile(r"[^a-z0-9_-]+")
_TOPIC_WS = re.compile(r"\s+")


class NoteParseError(ValueError):
    """Raised when a note response cannot become a usable grounded note."""


def synthesize_note(
    source_title: str,
    triage: Triage,
    items: list[KnowledgeItem],
    links: list[ApplicationLink],
    client: LLMClient,
) -> DistilledNote | None:
    """Build the reader-facing note from verified evidence, or fallback if the model fails."""
    if not items:
        return None

    try:
        raw = client.complete(build_note_prompt(source_title, triage, items, links), system=SYSTEM)
        return _parse_note(raw, items, links)
    except Exception:
        return _fallback_note(items, links)


def _parse_note(
    raw: str,
    items: list[KnowledgeItem],
    links: list[ApplicationLink],
) -> DistilledNote:
    data = _parse_object(raw)
    try:
        note = DistilledNote.model_validate(data)
    except ValidationError as exc:
        raise NoteParseError(f"Note JSON did not match schema: {exc}") from exc
    cleaned = _clean_note(note, items, links)
    if cleaned is None:
        raise NoteParseError("Note had no usable grounded core takeaway.")
    return cleaned


def _parse_object(raw: str) -> dict[str, Any]:
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise NoteParseError(f"Note response was not JSON: {raw[:120]!r}") from exc
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            raise NoteParseError(f"Note response was not JSON: {raw[:120]!r}") from exc2
    if not isinstance(data, dict):
        raise NoteParseError("Note response must be a JSON object.")
    return data


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped


def _clean_note(
    note: DistilledNote,
    items: list[KnowledgeItem],
    links: list[ApplicationLink],
) -> DistilledNote | None:
    valid_items = {item.item_id for item in items}
    valid_links = {link.link_id for link in links}

    core = _clean_grounded(note.core_takeaway, valid_items)
    if core is None:
        return None

    key_points = _clean_grounded_list(note.key_points, valid_items)
    why_it_matters = _clean_grounded_list(note.why_it_matters, valid_items)
    caveats = _clean_grounded_list(note.caveats, valid_items)
    how_to_apply = _clean_action_list(note.how_to_apply, valid_items, valid_links)
    review_questions = _clean_question_list(note.review_questions, valid_items)

    return DistilledNote(
        title=note.title.strip(),
        core_takeaway=core,
        key_points=key_points,
        why_it_matters=why_it_matters,
        how_to_apply=how_to_apply,
        caveats=caveats,
        review_questions=review_questions,
        topics=_normalize_topics(note.topics),
        generated_from="llm",
    )


def _clean_grounded(section: GroundedText, valid_items: set[str]) -> GroundedText | None:
    text = section.text.strip()
    item_ids = _valid_ordered(section.item_ids, valid_items)
    if not text or not item_ids:
        return None
    return GroundedText(text=text, item_ids=item_ids)


def _clean_grounded_list(
    sections: list[GroundedText],
    valid_items: set[str],
) -> list[GroundedText]:
    cleaned: list[GroundedText] = []
    for section in sections:
        item = _clean_grounded(section, valid_items)
        if item is not None:
            cleaned.append(item)
    return cleaned


def _clean_action_list(
    sections: list[ActionStep],
    valid_items: set[str],
    valid_links: set[str],
) -> list[ActionStep]:
    cleaned: list[ActionStep] = []
    for section in sections:
        text = section.text.strip()
        item_ids = _valid_ordered(section.item_ids, valid_items)
        if not text or not item_ids:
            continue
        cleaned.append(
            ActionStep(
                text=text,
                item_ids=item_ids,
                application_link_ids=_valid_ordered(section.application_link_ids, valid_links),
            )
        )
    return cleaned


def _clean_question_list(
    sections: list[ReviewQuestion],
    valid_items: set[str],
) -> list[ReviewQuestion]:
    cleaned: list[ReviewQuestion] = []
    for section in sections:
        question = section.question.strip()
        item_ids = _valid_ordered(section.item_ids, valid_items)
        if question and item_ids:
            cleaned.append(ReviewQuestion(question=question, item_ids=item_ids))
    return cleaned


def _valid_ordered(ids: list[str], valid: set[str]) -> list[str]:
    out: list[str] = []
    for item_id in ids:
        if item_id in valid and item_id not in out:
            out.append(item_id)
    return out


def _normalize_topics(topics: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw in topics:
        topic = _TOPIC_WS.sub("_", raw.strip().lower())
        topic = _TOPIC_CHARS.sub("", topic).strip("_-")
        if topic and topic not in normalized:
            normalized.append(topic)
        if len(normalized) >= 8:
            break
    return normalized


def _fallback_note(items: list[KnowledgeItem], links: list[ApplicationLink]) -> DistilledNote | None:
    if not items:
        return None
    first = items[0]
    key_points = [
        GroundedText(text=item.statement, item_ids=[item.item_id])
        for item in items[:8]
    ]
    actions = [
        ActionStep(
            text=link.scenario,
            item_ids=_valid_ordered(link.knowledge_item_ids, {item.item_id for item in items}),
            application_link_ids=[link.link_id],
        )
        for link in links
        if _valid_ordered(link.knowledge_item_ids, {item.item_id for item in items})
    ]

    caveats: list[GroundedText] = []
    for item in items:
        details: list[str] = []
        if item.scope:
            details.append(f"Scope: {item.scope}")
        if item.gotchas:
            details.append("Gotchas: " + "; ".join(item.gotchas))
        if item.speaker_confidence == "low":
            details.append("Speaker confidence is low.")
        if details:
            caveats.append(GroundedText(text=" ".join(details), item_ids=[item.item_id]))

    questions = [
        ReviewQuestion(
            question=f"How would you use this idea: {item.statement}",
            item_ids=[item.item_id],
        )
        for item in items[:3]
    ]

    return DistilledNote(
        title="",
        core_takeaway=GroundedText(text=first.statement, item_ids=[first.item_id]),
        key_points=key_points,
        how_to_apply=actions,
        caveats=caveats,
        review_questions=questions,
        topics=[],
        generated_from="fallback",
    )
