"""Phase D — entities ride along in the existing extraction call. No second transcript pass:
entities are a nested ``entities`` array on each knowledge item's JSON object, cleaned in code
(``extract._clean_entity_mentions``) before the item is ever validated, so a malformed entity
can never drop the knowledge item beside it.
"""

import json

import pytest

from distil.extract import run_extraction
from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.models import Triage


def _triage(dominant: str = "conceptual") -> Triage:
    return Triage.model_validate(
        {
            "knowledge_types_present": [{"type": dominant, "share": 0.9}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        }
    )


def _item(**overrides) -> dict:
    base = {
        "type": "conceptual",
        "statement": "React uses a virtual DOM.",
        "stance": "fact",
        "speaker_confidence": "high",
        "provenance": {"quote": "react uses a virtual dom", "timestamp": None, "locator": None},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_entities_come_back_from_the_same_extraction_call():
    """A single FakeClient response (one call) carries both the knowledge item and its entity —
    proof there is no second transcript pass."""
    t = ingest_text("react uses a virtual dom under the hood")
    resp = json.dumps([
        _item(entities=[
            {"name": "React", "kind": "tool", "description": "A JS UI library.",
             "quote": "react uses a virtual dom", "timestamp": None}
        ])
    ])
    fake = FakeClient(responses=[resp])
    items = run_extraction(t, _triage(), fake)

    assert fake.call_count == 1
    assert len(items) == 1
    assert len(items[0].entity_mentions) == 1
    mention = items[0].entity_mentions[0]
    assert mention.name == "React"
    assert mention.kind == "tool"
    assert mention.quote == "react uses a virtual dom"


@pytest.mark.unit
def test_item_with_no_entities_gets_empty_list():
    t = ingest_text("keep functions small and focused")
    resp = json.dumps([_item(statement="Keep functions small.", provenance={
        "quote": "keep functions small", "timestamp": None, "locator": None,
    })])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert items[0].entity_mentions == []


@pytest.mark.unit
def test_malformed_entity_is_dropped_without_harming_the_knowledge_item():
    """An entity with an invalid `kind` (outside tool/person/organization) is dropped — the
    item it rode in on still comes back whole."""
    t = ingest_text("react uses a virtual dom under the hood")
    resp = json.dumps([
        _item(entities=[
            {"name": "React", "kind": "framework", "description": "bad kind",
             "quote": "react uses a virtual dom", "timestamp": None}
        ])
    ])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert items[0].statement == "React uses a virtual DOM."
    assert items[0].entity_mentions == []


@pytest.mark.unit
def test_entity_missing_name_is_dropped():
    t = ingest_text("react uses a virtual dom under the hood")
    resp = json.dumps([
        _item(entities=[{"kind": "tool", "quote": "react uses a virtual dom"}])
    ])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert items[0].entity_mentions == []


@pytest.mark.unit
def test_one_bad_entity_does_not_drop_its_valid_sibling():
    t = ingest_text("react and openai are both mentioned here today")
    resp = json.dumps([
        _item(entities=[
            {"name": "React", "kind": "tool", "quote": "react", "timestamp": None},
            {"name": "Bad", "kind": "not_a_kind", "quote": "bad", "timestamp": None},
        ])
    ])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert [m.name for m in items[0].entity_mentions] == ["React"]


@pytest.mark.unit
def test_entity_quote_over_word_limit_is_truncated_not_dropped():
    long_quote = " ".join(["word"] * 20)
    t = ingest_text(long_quote + " and more text")
    resp = json.dumps([
        _item(entities=[
            {"name": "Thing", "kind": "tool", "quote": long_quote, "timestamp": None}
        ])
    ])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert len(items[0].entity_mentions) == 1
    assert len(items[0].entity_mentions[0].quote.split()) <= 14


@pytest.mark.unit
def test_entities_array_itself_malformed_still_returns_the_item():
    """A wholesale-broken `entities` payload (not a list at all) degrades to an empty list —
    it never fails or drops the knowledge item."""
    t = ingest_text("react uses a virtual dom under the hood")
    resp = json.dumps([_item(entities="not-a-list")])
    items = run_extraction(t, _triage(), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert items[0].entity_mentions == []


@pytest.mark.unit
def test_salvage_floor_for_items_is_unaffected_by_entities():
    """The existing item-level salvage floor (T-E7/T-E8) still fires exactly as before; entities
    have no floor of their own and never influence it."""
    from distil.triage import ParseError

    bad_item = {"type": "personal_experience", "statement": "missing stance and provenance",
                "entities": [{"name": "X", "kind": "tool", "quote": "x"}]}
    resp = json.dumps([_item(), bad_item, bad_item, bad_item])
    t = ingest_text("some transcript text about a concept")
    with pytest.raises(ParseError):
        run_extraction(t, _triage(), FakeClient(responses=[resp]))
