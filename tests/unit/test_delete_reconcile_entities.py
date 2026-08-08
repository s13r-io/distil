"""Phase D — delete cascade and bundle reconcile also cover entities/, mirroring the concept
gaps ``run_delete_entry_stage``/``reconcile_okf_bundle`` already close (see
test_delete_cascade.py / test_reconcile.py for the concept originals).
"""

import json

import pytest

from distil.canonicalize import canonicalize_entry_entities, run_delete_entry_stage
from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.okf_lint import lint
from distil.reconcile import reconcile_okf_bundle
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


def _mention(name="React", kind="tool", quote="react"):
    return {"name": name, "kind": kind, "description": "A JS UI library.", "quote": quote, "timestamp": None}


def _entry(entry_id, title, items) -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": items,
            "tags": {"topics": ["frontend"], "knowledge_types": ["conceptual"], "application_forms": []},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _transcript() -> Transcript:
    return Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp="00:00:00")])


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


@pytest.mark.unit
def test_sole_source_deletion_removes_orphaned_entity_page(store, embedder):
    e1 = _entry("e_01", "Intro to React", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    canonicalize_entry_entities(
        e1, store,
        FakeClient(responses=[json.dumps([
            {"mention_key": "k_01#0", "decision": "new", "title": "React", "description": "d"}
        ])]),
    )
    from distil import okf as okf_mod

    entity_id = store.list_entities()[0].entity_id
    okf_mod.export_entity(store.load_entity(entity_id), store, store.okf_root)
    okf_mod.render_source_with_concepts(e1, store, store.okf_root)
    entity_page = store.okf_root / "entities" / f"{entity_id}.md"
    assert entity_page.exists()

    run_delete_entry_stage("e_01", store)

    assert store.load_entity(entity_id) is None
    assert not entity_page.exists()
    entities_index = (store.okf_root / "entities" / "index.md").read_text()
    assert entity_id not in entities_index
    assert lint(store.okf_root) == []


@pytest.mark.unit
def test_surviving_entity_page_drops_stale_backreference_on_delete(store, embedder):
    e1 = _entry("e_01", "Intro to React", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention(quote="a")])])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    canonicalize_entry_entities(
        e1, store,
        FakeClient(responses=[json.dumps([
            {"mention_key": "k_01#0", "decision": "new", "title": "React", "description": "d"}
        ])]),
    )
    from distil import okf as okf_mod

    entity_id = store.list_entities()[0].entity_id
    okf_mod.export_entity(store.load_entity(entity_id), store, store.okf_root)
    okf_mod.render_source_with_concepts(e1, store, store.okf_root)

    e2 = _entry("e_02", "React Deep Dive", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention(quote="b")])])
    store.file_entry(e2, embedder=embedder, transcript=_transcript())
    canonicalize_entry_entities(
        e2, store,
        FakeClient(responses=[json.dumps([
            {"mention_key": "k_01#0", "decision": "match", "entity_id": entity_id}
        ])]),
    )
    okf_mod.export_entity(store.load_entity(entity_id), store, store.okf_root)
    okf_mod.render_source_with_concepts(e2, store, store.okf_root)
    assert len(store.load_entity(entity_id).members) == 2

    run_delete_entry_stage("e_01", store)

    survivor = store.load_entity(entity_id)
    assert survivor is not None
    assert {m.entry_id for m in survivor.members} == {"e_02"}
    entity_page = (store.okf_root / "entities" / f"{entity_id}.md").read_text()
    assert "intro-to-react" not in entity_page
    assert "react-deep-dive" in entity_page
    assert lint(store.okf_root) == []


@pytest.mark.unit
def test_reconcile_removes_orphaned_entity_page(store, embedder):
    e1 = _entry("e_01", "Intro to React", [_item("k_01", _REACT_STATEMENT, entity_mentions=[_mention()])])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    canonicalize_entry_entities(
        e1, store,
        FakeClient(responses=[json.dumps([
            {"mention_key": "k_01#0", "decision": "new", "title": "React", "description": "d"}
        ])]),
    )
    from distil import okf as okf_mod

    entity_id = store.list_entities()[0].entity_id
    okf_mod.export_entity(store.load_entity(entity_id), store, store.okf_root)

    # Simulate drift: the DB row is gone but the OKF page it left behind survives.
    store.delete_entity(entity_id)
    entity_page = store.okf_root / "entities" / f"{entity_id}.md"
    assert entity_page.exists()

    report = reconcile_okf_bundle(store, apply=True)

    assert not entity_page.exists()
    assert f"entities/{entity_id}.md" in report.removed
