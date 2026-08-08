"""Surfacing the concept layer (Phase B) in the web app: a concepts list page, a concept
detail page with claims/sources/typed relationships/contradiction flag, entry-page links to
the raw transcript and contributed concepts, and a streamed bundle export.

Everything here is read-only over an already-written bundle — no ingest, no LLM calls.
"""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from distil.ingest import Segment, Transcript
from distil.models import Concept, ConceptClaim, ConceptEdge, ConceptMember, KBEntry, Profile
from distil.store import Store
from web.app import create_app


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    s = Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")
    s.save_profile(Profile(user_id="owner"))
    return tmp_path


def _file_entry(store: Store, entry_id: str, title: str, stance: str = "opinion") -> KBEntry:
    entry = KBEntry.model_validate({
        "entry_id": entry_id,
        "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": f"{entry_id}_k1",
            "type": "heuristic",
            "statement": "Write tests first.",
            "stance": stance,
            "provenance": {"quote": "write tests first", "timestamp": "00:01:00"},
        }],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    })
    transcript = Transcript(
        segments=[Segment(text="write tests first", locator="seg:0", timestamp="00:01:00")]
    )
    store.file_entry(entry, transcript=transcript)
    return entry


# ---- concepts list page -----------------------------------------------------------------


@pytest.mark.unit
def test_concepts_list_empty_bundle_shows_empty_state(seeded):
    client = TestClient(create_app())
    r = client.get("/concepts")
    assert r.status_code == 200
    assert "No concepts yet" in r.text


@pytest.mark.unit
def test_concepts_list_shows_title_description_and_video_count(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Talk About Testing")
    store.save_concept(Concept(
        concept_id="testing-first",
        title="Testing First",
        description="Write tests before code.",
        members=[ConceptMember(entry_id="e1", item_id="e1_k1", quote="write tests first",
                                timestamp="00:01:00")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    client = TestClient(create_app())
    r = client.get("/concepts")
    assert r.status_code == 200
    assert "Testing First" in r.text
    assert "Write tests before code." in r.text
    assert '"video_count": 1' in r.text
    assert "1 concept" in r.text


@pytest.mark.unit
def test_concepts_list_has_filter_box_and_sort_with_no_letter_bar(seeded):
    client = TestClient(create_app())
    r = client.get("/concepts")
    assert 'id="search"' in r.text
    assert "Filter by title or description" in r.text
    assert 'id="sort"' in r.text
    assert "Most videos first" in r.text
    assert "Alphabetical" in r.text
    # No A-Z letter-bar index — explicitly rejected in favor of the filter box.
    assert "letter-bar" not in r.text
    assert "az-index" not in r.text


# ---- concept detail page ----------------------------------------------------------------


@pytest.mark.unit
def test_concept_detail_missing_returns_404(seeded):
    client = TestClient(create_app())
    r = client.get("/concepts/does-not-exist")
    assert r.status_code == 404


@pytest.mark.unit
def test_concept_detail_shows_claims_sources_and_edges(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Talk About Testing")
    store.save_concept(Concept(
        concept_id="pairing",
        title="Pairing",
        description="Two people, one keyboard.",
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    store.save_concept(Concept(
        concept_id="testing-first",
        title="Testing First",
        description="Write tests before code.",
        members=[ConceptMember(entry_id="e1", item_id="e1_k1", quote="write tests first",
                                timestamp="00:01:00")],
        claims=[ConceptClaim(text="Writing tests first improves design.", item_ids=["e1_k1"])],
        edges=[ConceptEdge(target_concept_id="pairing", relation="related")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    client = TestClient(create_app())
    r = client.get("/concepts/testing-first")
    assert r.status_code == 200
    assert "Writing tests first improves design." in r.text
    assert 'href="/entries/e1#e1_k1"' in r.text  # claim citation links to the entry + item
    assert 'href="/entries/e1"' in r.text  # sources section
    assert "write tests first" in r.text
    assert 'href="/concepts/pairing"' in r.text  # typed relationship link
    assert "Related" in r.text


@pytest.mark.unit
def test_concept_detail_flags_contradiction_between_members(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Pro TDD Talk", stance="opinion")
    _file_entry(store, "e2", "Anti TDD Talk", stance="fact")
    store.save_concept(Concept(
        concept_id="testing-first",
        title="Testing First",
        description="Write tests before code.",
        members=[
            ConceptMember(entry_id="e1", item_id="e1_k1", quote="write tests first"),
            ConceptMember(entry_id="e2", item_id="e2_k1", quote="write tests first"),
        ],
        claims=[ConceptClaim(text="Opinions differ.", item_ids=["e1_k1", "e2_k1"])],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    client = TestClient(create_app())
    r = client.get("/concepts/testing-first")
    assert r.status_code == 200
    assert "Contradiction" in r.text
    assert "disagree on stance" in r.text


@pytest.mark.unit
def test_concept_detail_edges_dont_include_dangling_targets(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.save_concept(Concept(
        concept_id="orphaned-edge",
        title="Has Dangling Edge",
        description="Edge points at a concept that no longer exists.",
        edges=[ConceptEdge(target_concept_id="ghost", relation="builds_on")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    client = TestClient(create_app())
    r = client.get("/concepts/orphaned-edge")
    assert r.status_code == 200
    assert "No typed relationships yet" in r.text


# ---- entry page additions ---------------------------------------------------------------


@pytest.mark.unit
def test_entry_page_links_transcript_and_contributed_concepts(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Talk About Testing")
    store.save_concept(Concept(
        concept_id="testing-first",
        title="Testing First",
        description="Write tests before code.",
        members=[ConceptMember(entry_id="e1", item_id="e1_k1", quote="write tests first")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    ))
    client = TestClient(create_app())
    r = client.get("/entries/e1")
    assert r.status_code == 200
    assert 'href="/entries/e1/transcript.md"' in r.text
    assert 'href="/concepts/testing-first"' in r.text
    assert "Testing First" in r.text


@pytest.mark.unit
def test_entry_page_hides_transcript_link_when_no_bundle_exported(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.file_entry(KBEntry.model_validate({
        "entry_id": "e_no_transcript",
        "source": {"title": "No transcript export", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }))  # no transcript= kwarg -> no OKF export
    client = TestClient(create_app())
    r = client.get("/entries/e_no_transcript")
    assert r.status_code == 200
    assert "transcript.md" not in r.text


@pytest.mark.unit
def test_transcript_route_serves_raw_markdown(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Talk About Testing")
    client = TestClient(create_app())
    r = client.get("/entries/e1/transcript.md")
    assert r.status_code == 200
    assert "write tests first" in r.text


@pytest.mark.unit
def test_transcript_route_404s_when_not_exported(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.file_entry(KBEntry.model_validate({
        "entry_id": "e_no_transcript",
        "source": {"title": "No transcript export", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }))
    client = TestClient(create_app())
    r = client.get("/entries/e_no_transcript/transcript.md")
    assert r.status_code == 404


@pytest.mark.unit
def test_transcript_route_404s_for_unknown_entry(seeded):
    client = TestClient(create_app())
    r = client.get("/entries/does-not-exist/transcript.md")
    assert r.status_code == 404


# ---- bundle export -----------------------------------------------------------------------


@pytest.mark.unit
def test_bundle_download_streams_a_valid_zip_of_the_okf_bundle(seeded):
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    _file_entry(store, "e1", "Talk About Testing")
    client = TestClient(create_app())
    r = client.get("/bundle.zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="distil-bundle.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "sources/talk-about-testing.md" in names
    assert "raw/talk-about-testing.md" in names
    assert "index.md" in names
    # nothing outside the bundle directory
    assert all(not name.startswith("..") for name in names)


@pytest.mark.unit
def test_bundle_download_on_empty_or_missing_bundle_is_a_valid_empty_zip(seeded):
    client = TestClient(create_app())
    r = client.get("/bundle.zip")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.namelist() == []
