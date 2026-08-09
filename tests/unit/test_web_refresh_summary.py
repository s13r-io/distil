"""POST /entries/{id}/refresh-summary — the web counterpart to `distil refresh-summary`.

Generates/regenerates only the narrative summary from the entry's stored raw transcript; never
re-fetches, never re-extracts. Uses the FakeClient seam, no real model calls.
"""

import pytest
from fastapi.testclient import TestClient

from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry, Profile
from distil.store import Store
from web import app as webapp
from web.app import create_app


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    store = Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")
    store.save_profile(Profile(user_id="owner"))
    return store


def _file_entry(store: Store, with_transcript: bool = True) -> KBEntry:
    entry = KBEntry.model_validate({
        "entry_id": "e_01",
        "source": {"title": "A talk", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01", "type": "heuristic", "statement": "Keep functions small.",
            "stance": "opinion", "provenance": {"quote": "keep functions small"},
        }],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    })
    if with_transcript:
        transcript = Transcript(segments=[
            Segment(text="Keep functions small and focused on one job.", locator="seg:0"),
            Segment(text="It makes testing dramatically easier down the line.", locator="seg:1"),
        ])
        store.file_entry(entry, transcript=transcript)
    else:
        store.file_entry(entry)
    return entry


@pytest.mark.unit
def test_refresh_summary_route_regenerates_summary(seeded, monkeypatch):
    _file_entry(seeded)
    monkeypatch.setattr(
        webapp, "_make_summary_client", lambda: FakeClient(responses=["N" * 200])
    )
    client = TestClient(create_app())

    r = client.post("/entries/e_01/refresh-summary")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    reloaded = seeded.load_entry("e_01")
    assert reloaded.narrative_summary is not None
    assert reloaded.narrative_summary.text == "N" * 200


@pytest.mark.unit
def test_refresh_summary_route_missing_entry_404s(seeded, monkeypatch):
    monkeypatch.setattr(webapp, "_make_summary_client", lambda: FakeClient(responses=[]))
    client = TestClient(create_app())
    r = client.post("/entries/e_missing/refresh-summary")
    assert r.status_code == 404


@pytest.mark.unit
def test_refresh_summary_route_reports_missing_transcript_as_ok_false(seeded, monkeypatch):
    _file_entry(seeded, with_transcript=False)
    monkeypatch.setattr(webapp, "_make_summary_client", lambda: FakeClient(responses=[]))
    client = TestClient(create_app())

    r = client.post("/entries/e_01/refresh-summary")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "transcript" in body["message"].lower()


@pytest.mark.unit
def test_entry_page_shows_narrative_summary_when_present(seeded, monkeypatch):
    entry = _file_entry(seeded)
    entry.narrative_summary = None
    monkeypatch.setattr(
        webapp, "_make_summary_client", lambda: FakeClient(responses=["N" * 200])
    )
    client = TestClient(create_app())
    client.post("/entries/e_01/refresh-summary")

    r = client.get("/entries/e_01")
    assert r.status_code == 200
    assert "Narrative summary" in r.text
    assert "N" * 200 in r.text
