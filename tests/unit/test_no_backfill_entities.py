"""Phase D — NO BACKFILL: entities apply only to newly ingested videos. An existing entry
(filed before this phase, or simply one whose extraction found no named entities) carries no
entity data, and every surface must degrade gracefully rather than error.
"""

import pytest

from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.models import KBEntry, KnowledgeItem
from distil.okf_lint import lint
from distil.store import Store


@pytest.mark.unit
def test_pre_phase_d_kb_json_without_entity_mentions_field_parses_cleanly():
    """A kb/<id>.md front-matter payload written before entity_mentions existed (the field is
    simply absent from the JSON) must still validate — this is exactly what an old, un-migrated
    entry looks like on disk."""
    item = KnowledgeItem.model_validate({
        "item_id": "k_01",
        "type": "heuristic",
        "statement": "Keep functions small.",
        "stance": "opinion",
        "provenance": {"quote": "keep functions small"},
    })
    assert item.entity_mentions == []


def _entry(entry_id: str, title: str) -> KBEntry:
    return KBEntry.model_validate({
        "entry_id": entry_id,
        "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01",
            "type": "heuristic",
            "statement": "Keep functions small.",
            "stance": "opinion",
            "provenance": {"quote": "keep functions small"},
        }],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    })


@pytest.mark.unit
def test_entry_with_no_entities_files_and_exports_cleanly(tmp_path):
    store = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")
    entry = _entry("e_old", "An Old Video")
    store.file_entry(
        entry, embedder=FakeEmbedder(dim=16),
        transcript=Transcript(segments=[Segment(text="keep functions small", locator="seg:0")]),
    )
    assert store.load_entity("anything") is None
    assert store.list_entities() == []
    # A bundle with zero entities still lints clean — an empty entities/index.md is normal.
    assert lint(store.okf_root) == []


@pytest.mark.unit
def test_web_entities_list_renders_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    from fastapi.testclient import TestClient

    from distil.models import Profile
    from web.app import create_app

    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(Profile(user_id="owner"))
    client = TestClient(create_app())
    r = client.get("/entities")
    assert r.status_code == 200
    assert "No entities yet" in r.text


@pytest.mark.unit
def test_web_entry_page_with_no_entities_renders_without_entities_section(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    from fastapi.testclient import TestClient

    from distil.models import Profile
    from web.app import create_app

    store = Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")
    store.save_profile(Profile(user_id="owner"))
    store.file_entry(_entry("e_old", "An Old Video"))

    client = TestClient(create_app())
    r = client.get("/entries/e_old")
    assert r.status_code == 200
    assert "An Old Video" in r.text


@pytest.mark.unit
def test_entity_detail_route_404s_for_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    from fastapi.testclient import TestClient

    from distil.models import Profile
    from web.app import create_app

    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(Profile(user_id="owner"))
    client = TestClient(create_app())
    r = client.get("/entities/does-not-exist")
    assert r.status_code == 404
