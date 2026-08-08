"""Phase D — canonicalize.py mirrors its Concept match/new/reject/synthesis shape for entities,
one granularity down (entity mentions instead of knowledge items). Same stage as concepts
(canonicalize.py), same per-video capping discipline — no new pipeline stage.
"""

import json

import pytest

from distil.canonicalize import (
    MAX_ENTITIES_TO_SYNTHESIZE_PER_VIDEO,
    canonicalize_entry_entities,
    synthesize_touched_entities,
)
from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Entity, EntityMember, KBEntry
from distil.store import Store

_REACT_STATEMENT = "React renders UI using a virtual DOM diffing algorithm"


def _item(item_id, statement, *, entity_mentions=None, quote="q"):
    return {
        "item_id": item_id,
        "type": "conceptual",
        "statement": statement,
        "stance": "fact",
        "provenance": {"quote": quote},
        "entity_mentions": entity_mentions or [],
    }


def _mention(name="React", kind="tool", description="A JS UI library.", quote="react"):
    return {"name": name, "kind": kind, "description": description, "quote": quote, "timestamp": None}


def _entry(entry_id, items, topics=("frontend",)) -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": f"Video {entry_id}", "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": items,
            "tags": {"topics": list(topics), "knowledge_types": ["conceptual"], "application_forms": []},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


@pytest.mark.unit
def test_no_mentions_no_llm_call_no_entities(store, embedder):
    e1 = _entry("e_01", [_item("k_01", "Keep functions small.")])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(responses=["should-not-be-used"])
    touched = canonicalize_entry_entities(e1, store, fake)
    assert touched == []
    assert fake.call_count == 0


@pytest.mark.unit
def test_new_entity_created_from_a_mention(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(responses=[
        json.dumps([
            {"mention_key": "k_01#0", "decision": "new", "title": "React",
             "description": "A JS UI library."}
        ])
    ])
    touched = canonicalize_entry_entities(e1, store, fake)
    assert len(touched) == 1
    entity = touched[0]
    assert entity.kind == "tool"
    assert entity.title == "React"
    assert len(entity.members) == 1
    assert entity.members[0] == EntityMember(
        entry_id="e_01", item_id="k_01", quote="react", timestamp=None
    )


@pytest.mark.unit
def test_same_entity_across_two_videos_merges_into_one_page_with_two_members(store, embedder):
    """The core acceptance criterion: two videos mentioning the same tool merge into one Entity
    with two members, rather than two separate entity pages."""
    e1 = _entry("e_01", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention(quote="react a")])])
    store.file_entry(e1, embedder=embedder)
    entity = Entity(
        entity_id="react", kind="tool", title="React", description="A JS UI library.",
        members=[EntityMember(entry_id="e_01", item_id="k_01", quote="react a", timestamp=None)],
        created_at="2026-06-15T00:00:00", updated_at="2026-06-15T00:00:00",
    )
    store.save_entity(entity)

    e2 = _entry("e_02", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention(quote="react b")])])
    store.file_entry(e2, embedder=embedder)
    fake = FakeClient(responses=[
        json.dumps([{"mention_key": "k_01#0", "decision": "match", "entity_id": "react"}])
    ])
    touched = canonicalize_entry_entities(e2, store, fake)

    assert len(touched) == 1
    merged = store.load_entity("react")
    assert merged is not None
    assert {m.entry_id for m in merged.members} == {"e_01", "e_02"}
    assert len(merged.members) == 2


@pytest.mark.unit
def test_reject_produces_no_entity(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(responses=[json.dumps([{"mention_key": "k_01#0", "decision": "reject"}])])
    touched = canonicalize_entry_entities(e1, store, fake)
    assert touched == []
    assert store.list_entities() == []


@pytest.mark.unit
def test_match_to_untrusted_entity_id_is_ignored(store, embedder):
    """A decision that matches to an entity_id never offered as a candidate is not trusted —
    treated as no record, same discipline as concept canonicalize's untrusted-concept-id path."""
    e1 = _entry("e_01", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(responses=[
        json.dumps([{"mention_key": "k_01#0", "decision": "match", "entity_id": "hallucinated"}])
    ])
    touched = canonicalize_entry_entities(e1, store, fake)
    assert touched == []


@pytest.mark.unit
def test_reprocessing_the_same_entry_is_idempotent(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder)
    resp = json.dumps([
        {"mention_key": "k_01#0", "decision": "new", "title": "React", "description": "d"}
    ])
    canonicalize_entry_entities(e1, store, FakeClient(responses=[resp]))
    entity_id = store.list_entities()[0].entity_id

    canonicalize_entry_entities(e1, store, FakeClient(responses=[resp]))
    assert len(store.list_entities()) == 1
    assert len(store.load_entity(entity_id).members) == 1


@pytest.mark.unit
def test_synthesis_capping_marks_excess_pending(store, embedder):
    items = []
    for i in range(1, 8):
        items.append(_item(f"k_{i:02d}", f"Fact number {i} about tools.",
                            entity_mentions=[_mention(name=f"Tool{i}", quote=f"tool {i}")]))
    e1 = _entry("e_01", items)
    store.file_entry(e1, embedder=embedder)

    decisions = [
        {"mention_key": f"k_{i:02d}#0", "decision": "new", "title": f"Tool{i}", "description": "d"}
        for i in range(1, 8)
    ]
    fake = FakeClient(responses=[json.dumps(decisions)])
    touched = canonicalize_entry_entities(e1, store, fake)
    assert len(touched) == 7

    synth_client = FakeClient(responses=[
        json.dumps([{"text": "Claim about tool.", "item_ids": [f"k_{i:02d}"]}])
        for i in range(MAX_ENTITIES_TO_SYNTHESIZE_PER_VIDEO)
    ])
    synthesize_touched_entities(e1, touched, store, synth_client)

    all_entities = store.list_entities()
    pending = [e for e in all_entities if e.pending_synthesis]
    synthesized = [e for e in all_entities if not e.pending_synthesis]
    assert len(pending) == 7 - MAX_ENTITIES_TO_SYNTHESIZE_PER_VIDEO
    assert len(synthesized) == MAX_ENTITIES_TO_SYNTHESIZE_PER_VIDEO
