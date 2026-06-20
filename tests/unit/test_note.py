"""Note v1 — reader-facing synthesis over verified items."""

import json

import pytest

from distil.llm import FakeClient
from distil.models import ApplicationLink, KnowledgeItem, Triage
from distil.note import synthesize_note


@pytest.fixture
def triage():
    return Triage.model_validate({
        "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
        "density": "high",
        "transcript_loss": {"level": "low", "evidence": []},
        "verdict": "rich",
    })


@pytest.fixture
def items():
    return [
        KnowledgeItem.model_validate({
            "item_id": "k_01",
            "type": "heuristic",
            "statement": "Keep functions small.",
            "stance": "opinion",
            "speaker_confidence": "high",
            "scope": "Library code.",
            "provenance": {"quote": "keep functions small"},
        }),
        KnowledgeItem.model_validate({
            "item_id": "k_02",
            "type": "declarative",
            "statement": "Small functions are easier to test.",
            "stance": "fact",
            "speaker_confidence": "medium",
            "provenance": {"quote": "easier to test"},
        }),
    ]


@pytest.fixture
def links():
    return [
        ApplicationLink.model_validate({
            "link_id": "a_01",
            "knowledge_item_ids": ["k_01"],
            "linked_goal_id": "g_01",
            "application_form": "checklist",
            "scenario": "Check auth helpers for functions doing more than one thing.",
        })
    ]


def _valid_note(**overrides):
    data = {
        "title": "Small Function Design",
        "core_takeaway": {"text": "Small focused functions are easier to improve.", "item_ids": ["k_01"]},
        "key_points": [{"text": "Keep each function focused.", "item_ids": ["k_01"]}],
        "why_it_matters": [{"text": "Testing becomes easier.", "item_ids": ["k_02"]}],
        "how_to_apply": [{
            "text": "Review auth helpers for mixed responsibilities.",
            "item_ids": ["k_01"],
            "application_link_ids": ["a_01"],
        }],
        "caveats": [{"text": "This applies most clearly to library code.", "item_ids": ["k_01"]}],
        "review_questions": [{"question": "Which function should you split?", "item_ids": ["k_01"]}],
        "topics": ["Function Design", " unit testing ", "Function Design", "bad topic!"],
    }
    data.update(overrides)
    return json.dumps(data)


@pytest.mark.unit
def test_note_parses_valid_grounded_json(triage, items, links):
    note = synthesize_note("Talk", triage, items, links, FakeClient([_valid_note()]))
    assert note.title == "Small Function Design"
    assert note.generated_from == "llm"
    assert note.core_takeaway.item_ids == ["k_01"]
    assert note.how_to_apply[0].application_link_ids == ["a_01"]


@pytest.mark.unit
def test_note_drops_unknown_item_and_link_refs(triage, items, links):
    raw = _valid_note(
        key_points=[
            {"text": "valid", "item_ids": ["k_01", "k_fake"]},
            {"text": "drop me", "item_ids": ["k_fake"]},
        ],
        how_to_apply=[
            {"text": "valid action", "item_ids": ["k_01"], "application_link_ids": ["a_fake"]},
            {"text": "drop action", "item_ids": ["k_fake"], "application_link_ids": ["a_01"]},
        ],
    )
    note = synthesize_note("Talk", triage, items, links, FakeClient([raw]))
    assert len(note.key_points) == 1
    assert note.key_points[0].item_ids == ["k_01"]
    assert len(note.how_to_apply) == 1
    assert note.how_to_apply[0].application_link_ids == []


@pytest.mark.unit
def test_note_normalizes_topics(triage, items, links):
    note = synthesize_note("Talk", triage, items, links, FakeClient([_valid_note()]))
    assert note.topics == ["function_design", "unit_testing", "bad_topic"]


@pytest.mark.unit
def test_note_falls_back_on_malformed_output(triage, items, links):
    note = synthesize_note("Talk", triage, items, links, FakeClient(["not json"]))
    assert note.generated_from == "fallback"
    assert note.core_takeaway.text == "Keep functions small."
    assert note.how_to_apply[0].application_link_ids == ["a_01"]


@pytest.mark.unit
def test_note_returns_none_for_empty_items(triage, links):
    note = synthesize_note("Talk", triage, [], links, FakeClient(["SHOULD NOT BE CALLED"]))
    assert note is None
