"""Phase 1.2 — store.py: SQLite index + markdown filing. Tests T-S1..S3."""

import pytest

from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.models import ActionStep, Concept, ConceptMember, GroundedText, KBEntry, ReviewQuestion
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
def test_new_note_entries_render_humanized_tags(store):
    entry = _entry(with_note=True)
    entry.distilled_note.topics = ["ai_agent-memory", "function_design"]
    path = store.file_entry(entry)
    text = path.read_text()
    body = text.split("---\n", 2)[-1]
    assert "*Tags:* AI Agent Memory, Function Design" in body
    assert "*Topics:*" not in body
    assert "ai_agent-memory" not in body


@pytest.mark.unit
def test_teaching_note_markdown_export_is_reader_facing(store):
    entry = _entry(with_note=True)
    entry.source.url = "https://www.youtube.com/watch?v=abc"
    entry.source.channel = "Talk Channel"
    note = entry.distilled_note
    note.topics = ["ai_agent-memory", "function_design"]
    note.why_it_matters = [GroundedText(text="Focused code is easier to review.", item_ids=["k_01"])]
    note.how_to_apply = [ActionStep(text="Split one large function this week.", item_ids=["k_01"])]
    note.caveats = [GroundedText(text="This is scoped to library code.", item_ids=["k_01"])]
    note.review_questions = [ReviewQuestion(question="Where is one function doing too much?",
                                            item_ids=["k_01"])]

    text = Store.teaching_note_markdown(entry)
    assert text.startswith("# Small functions")
    assert "## Metadata" in text
    assert "- Source URL: https://www.youtube.com/watch?v=abc" in text
    assert "## Core takeaway" in text
    assert "## Why it matters" in text
    assert "1. Split one large function this week." in text
    assert "## Review questions" in text
    assert "## Tags" in text
    assert "AI Agent Memory" in text
    assert "k_01" not in text


# ---- Thin-material advisory (visible, never a rejection) --------------------------------


@pytest.mark.unit
def test_thin_material_note_shown_when_word_count_is_low(store):
    entry = _entry(with_note=True)
    entry.source.transcript_word_count = 120
    path = store.file_entry(entry)
    text = path.read_text()
    assert "Thin material" in text
    assert "120" in text


@pytest.mark.unit
def test_thin_material_note_absent_when_word_count_is_healthy(store):
    entry = _entry(with_note=True)
    entry.source.transcript_word_count = 2000
    path = store.file_entry(entry)
    assert "Thin material" not in path.read_text()


@pytest.mark.unit
def test_thin_material_note_absent_for_unknown_zero_word_count(store):
    """word_count == 0 (entries filed before this field existed) is unknown, not thin."""
    entry = _entry(with_note=True)
    entry.source.transcript_word_count = 0
    path = store.file_entry(entry)
    assert "Thin material" not in path.read_text()


@pytest.mark.unit
def test_thin_material_note_shown_without_a_distilled_note_too(store):
    """The no-note fallback rendering path also carries the advisory."""
    entry = _entry(with_note=False)
    entry.source.transcript_word_count = 50
    path = store.file_entry(entry)
    assert "Thin material" in path.read_text()


@pytest.mark.unit
def test_thin_material_note_shown_in_teaching_note_export(store):
    entry = _entry(with_note=True)
    entry.source.transcript_word_count = 100
    text = Store.teaching_note_markdown(entry)
    assert "Thin material" in text
    assert "100" in text
    assert "---" not in text


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


@pytest.mark.unit
def test_list_entries_prunes_rows_whose_files_are_missing(store):
    entry = _entry()
    store.file_entry(entry)
    store.entry_path(entry.entry_id).unlink()
    assert store.list_entries() == []


@pytest.mark.unit
def test_list_entries_never_prunes_on_verdict_or_empty_items(store):
    """A filed entry with a little_to_extract verdict and zero knowledge items is a normal,
    expected outcome now (owner decision — pipeline.py's module docstring) — list_entries must
    never silently discard it. That would reintroduce, at listing time, exactly the silent
    quality-based discard the pipeline itself no longer does."""
    entry = _entry(entry_id="e_low", title="Low value upload")
    entry.triage.verdict = "little_to_extract"
    entry.knowledge_items = []
    store.file_entry(entry, embedder=FakeEmbedder(dim=8))

    assert store.entry_path("e_low").exists()
    assert [r.entry_id for r in store.list_entries()] == ["e_low"]
    assert store.entry_path("e_low").exists()


@pytest.mark.unit
def test_list_entries_prunes_rows_whose_file_is_missing(store):
    """Only genuinely stale index data (the kb/ file itself is gone) is pruned — a real
    filesystem/index mismatch, unrelated to triage verdict or item count."""
    entry = _entry(entry_id="e_gone", title="Will lose its file")
    store.file_entry(entry, embedder=FakeEmbedder(dim=8))
    store.entry_path("e_gone").unlink()

    assert store.list_entries() == []
    assert all(row.entry_id != "e_gone" for row in store.list_entries())


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


# ---- OKF export plumbing (Phase 2) -------------------------------------------------------


@pytest.mark.unit
def test_filing_with_transcript_exports_okf_pages(store):
    entry = _entry()
    transcript = Transcript(segments=[Segment(text="keep functions small", locator="seg:0",
                                              timestamp="00:12:30")])
    store.file_entry(entry, transcript=transcript)
    assert (store.okf_root / "sources" / "a-talk.md").exists()
    assert (store.okf_root / "raw" / "a-talk.md").exists()
    assert (store.okf_root / "index.md").exists()


@pytest.mark.unit
def test_filing_without_transcript_does_not_touch_okf(store):
    entry = _entry()
    store.file_entry(entry)  # e.g. a feedback-only re-file
    assert not store.okf_root.exists()


@pytest.mark.unit
def test_okf_root_defaults_to_sibling_of_kb_dir(tmp_path):
    store = Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")
    assert store.okf_root == tmp_path / "okf"


@pytest.mark.unit
def test_delete_entry_is_pure_db_and_does_not_touch_okf_pages(store):
    """Store.delete_entry is DB/file-store-only by design (see its docstring) — OKF page
    removal is orchestrated by canonicalize.run_delete_entry_stage instead (see
    tests/unit/test_delete_cascade.py), which needs the entry loaded before this call runs."""
    entry = _entry()
    transcript = Transcript(segments=[Segment(text="keep functions small", locator="seg:0",
                                              timestamp="00:12:30")])
    store.file_entry(entry, transcript=transcript)
    store.delete_entry("e_01")
    assert (store.okf_root / "sources" / "a-talk.md").exists()
    assert (store.okf_root / "raw" / "a-talk.md").exists()


# ---- Concepts table (Phase 15.1 — canonicalize engine, design report §3, §5) -------------


def _concept(concept_id="c1", title="C1", members=None) -> Concept:
    return Concept(
        concept_id=concept_id,
        title=title,
        description="d",
        members=members or [],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )


@pytest.mark.unit
def test_save_and_load_concept_round_trips(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    concept = _concept(
        members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)]
    )
    store.save_concept(concept)
    assert store.load_concept("c1") == concept
    assert [c.concept_id for c in store.list_concepts()] == ["c1"]


@pytest.mark.unit
def test_resaving_concept_updates_not_duplicates(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    store.save_concept(_concept(title="Old title"))
    store.save_concept(_concept(title="New title"))
    concepts = store.list_concepts()
    assert len(concepts) == 1
    assert concepts[0].title == "New title"


@pytest.mark.unit
def test_delete_concept(store):
    store.save_concept(_concept())
    assert store.delete_concept("c1") is True
    assert store.load_concept("c1") is None
    assert store.delete_concept("c1") is False


@pytest.mark.unit
def test_load_missing_concept_returns_none(store):
    assert store.load_concept("nope") is None


@pytest.mark.unit
def test_concept_centroid_is_mean_of_member_vectors_and_drives_candidates(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    vec = next(v for _iid, eid, v in store.iter_item_vectors() if eid == "e_01")
    store.save_concept(
        _concept(
            members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)]
        )
    )
    candidates = store.find_concept_candidates(vec, [], "")
    assert candidates and candidates[0].concept_id == "c1"
    assert candidates[0].similarity == pytest.approx(1.0)


@pytest.mark.unit
def test_concept_centroid_getter_matches_saved_centroid(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    vec = next(v for _iid, eid, v in store.iter_item_vectors() if eid == "e_01")
    store.save_concept(
        _concept(
            members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)]
        )
    )
    assert store.concept_centroid("c1") == pytest.approx(vec)


@pytest.mark.unit
def test_concept_centroid_missing_concept_returns_empty(store):
    assert store.concept_centroid("nope") == []


@pytest.mark.unit
def test_find_concept_candidates_below_floor_and_no_token_overlap_returns_empty(store):
    store.save_concept(_concept(title="Something Unrelated"))  # zero members -> empty centroid
    candidates = store.find_concept_candidates([1.0, 0.0], [], "totally different words")
    assert candidates == []


@pytest.mark.unit
def test_find_concept_candidates_respects_max_candidates_env(store, monkeypatch):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    vec = next(v for _iid, eid, v in store.iter_item_vectors() if eid == "e_01")
    for i in range(3):
        store.save_concept(
            _concept(
                concept_id=f"c{i}",
                members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)],
            )
        )
    monkeypatch.setenv("DISTIL_CONCEPT_MAX_CANDIDATES", "1")
    assert len(store.find_concept_candidates(vec, [], "")) == 1


@pytest.mark.unit
def test_retract_entry_concept_memberships_deletes_zero_member_concept(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    store.save_concept(
        _concept(
            members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)]
        )
    )
    store.retract_entry_concept_memberships("e_01")
    assert store.load_concept("c1") is None


@pytest.mark.unit
def test_retract_entry_concept_memberships_keeps_other_entries_members(store):
    store.file_entry(_entry(entry_id="e_01"), embedder=FakeEmbedder(dim=8))
    store.file_entry(_entry(entry_id="e_02"), embedder=FakeEmbedder(dim=8))
    store.save_concept(
        _concept(
            members=[
                ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None),
                ConceptMember(entry_id="e_02", item_id="k_01", quote="q", timestamp=None),
            ]
        )
    )
    store.retract_entry_concept_memberships("e_01")
    remaining = store.load_concept("c1")
    assert remaining is not None
    assert {m.entry_id for m in remaining.members} == {"e_02"}


@pytest.mark.unit
def test_delete_entry_retracts_concept_membership(store):
    store.file_entry(_entry(), embedder=FakeEmbedder(dim=8))
    store.save_concept(
        _concept(
            members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)]
        )
    )
    store.delete_entry("e_01")
    assert store.load_concept("c1") is None
