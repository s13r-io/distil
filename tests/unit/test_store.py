"""Phase 1.2 — store.py: SQLite index + markdown filing. Tests T-S1..S3."""

import pytest

from distil.embed import FakeEmbedder
from distil.models import KBEntry
from distil.store import Store


def _entry(
    entry_id: str = "e_01",
    title: str = "A talk",
    score: int | None = None,
    *,
    with_note: bool = False,
) -> KBEntry:
    data = {
        "entry_id": entry_id,
        "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [
            {
                "item_id": "k_01",
                "type": "heuristic",
                "statement": "Keep functions small.",
                "stance": "opinion",
                "provenance": {"quote": "keep functions small", "timestamp": "00:12:30"},
            }
        ],
        "tags": {"topics": ["python"], "knowledge_types": ["heuristic"], "application_forms": []},
        "feedback": {"score": score} if score is not None else {},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }
    if with_note:
        data["distilled_note"] = {
            "title": "Small functions",
            "core_takeaway": {
                "text": "Small functions are easier to understand.",
                "item_ids": ["k_01"],
            },
            "key_points": [{"text": "Keep one behavior per function.", "item_ids": ["k_01"]}],
            "topics": ["python"],
        }
    return KBEntry.model_validate(data)


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")


# ---- T-S1: filing writes kb/<id>.md with valid front-matter and human-readable body ----


@pytest.mark.unit
def test_file_writes_markdown_with_frontmatter_and_body(store, tmp_path):
    entry = _entry()
    path = store.file_entry(entry)
    assert path.exists()
    assert path.name == "e_01.md"
    text = path.read_text()
    # Front matter delimited by ---
    assert text.startswith("---\n")
    assert text.count("---\n") >= 2
    # Body is human-readable: contains the statement and the title
    assert "Keep functions small." in text
    assert "A talk" in text


@pytest.mark.unit
def test_front_matter_round_trips_to_entry(store):
    entry = _entry()
    store.file_entry(entry)
    reloaded = store.load_entry("e_01")
    assert reloaded == entry


@pytest.mark.unit
def test_new_note_entries_render_teaching_note_and_evidence(store):
    path = store.file_entry(_entry(with_note=True))
    text = path.read_text()
    assert "## Core takeaway" in text
    assert "Small functions are easier to understand." in text
    assert "<summary>Source evidence</summary>" in text
    assert "keep functions small" in text


@pytest.mark.unit
def test_new_note_entries_render_source_url_and_index_note_title(store):
    entry = _entry(title="[English] weird_file-name.srt", with_note=True)
    entry.source.url = "https://youtu.be/abc123"
    path = store.file_entry(entry)
    text = path.read_text()
    assert "Source:* [Watch on YouTube](https://youtu.be/abc123)" in text
    assert "# Small functions" in text
    assert store.list_entries()[0].title == "Small functions"


# ---- T-S2: index row inserted; re-filing same id updates, not duplicates ----


@pytest.mark.unit
def test_filing_inserts_index_row(store):
    store.file_entry(_entry())
    rows = store.list_entries()
    assert len(rows) == 1
    assert rows[0].entry_id == "e_01"
    assert rows[0].title == "A talk"
    assert "heuristic" in rows[0].knowledge_types


@pytest.mark.unit
def test_refiling_same_id_updates_not_duplicates(store):
    store.file_entry(_entry(title="Old title"))
    store.file_entry(_entry(title="New title", score=5))
    rows = store.list_entries()
    assert len(rows) == 1
    assert rows[0].title == "New title"
    assert rows[0].score == 5


# ---- T-S3: KB and DB survive process restart (persistence) ----


@pytest.mark.unit
def test_persistence_across_new_store_instances(tmp_path):
    db = tmp_path / "distil.db"
    kb = tmp_path / "kb"
    Store(db_path=db, kb_dir=kb).file_entry(_entry())
    # A fresh Store object simulates a process restart against the same files.
    store2 = Store(db_path=db, kb_dir=kb)
    rows = store2.list_entries()
    assert len(rows) == 1
    reloaded = store2.load_entry("e_01")
    assert reloaded.entry_id == "e_01"
    assert reloaded.knowledge_items[0].provenance.quote == "keep functions small"


# ---- Profile persistence (used by score/link stages) ----


@pytest.mark.unit
def test_profile_save_and_load(store):
    from distil.models import Profile

    p = Profile(user_id="owner")
    store.save_profile(p)
    loaded = store.load_profile("owner")
    assert loaded == p


@pytest.mark.unit
def test_load_missing_profile_returns_none(store):
    assert store.load_profile("nobody") is None


@pytest.mark.unit
def test_candidate_lookup_by_topic(store):
    store.file_entry(_entry(entry_id="e_01"))
    store.file_entry(_entry(entry_id="e_02"))
    # both tagged python/heuristic; lookup excludes the query entry itself
    candidates = store.find_candidates(topics=["python"], knowledge_types=["heuristic"], exclude="e_01")
    ids = {c.entry_id for c in candidates}
    assert ids == {"e_02"}


@pytest.mark.unit
def test_delete_entry_removes_file_index_and_vectors(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    assert store.entry_path("e_01").exists()
    assert store.vector_count() == 1
    assert store.delete_entry("e_01") is True
    assert not store.entry_path("e_01").exists()
    assert store.list_entries() == []
    assert store.vector_count() == 0


@pytest.mark.unit
def test_delete_missing_entry_returns_false(store):
    assert store.delete_entry("e_missing") is False
