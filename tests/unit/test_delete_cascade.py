"""Delete-cascade orchestration (canonicalize.run_delete_entry_stage) — closes three gaps a
code review found in the merged delete path: orphaned concept pages, stale back-references on
surviving concepts, and unrecoverable raw/source pages when the kb file is missing or
unparseable. See the task brief this closes for the full gap analysis.
"""

import json

import pytest

from distil.canonicalize import run_canonicalize_stage, run_delete_entry_stage
from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.okf_lint import lint
from distil.store import Store

_AGENTIC_RAG = "Agentic RAG adds a planning loop before retrieval"


def _item(item_id, statement, *, quote="q", stance="fact", type_="conceptual"):
    return {
        "item_id": item_id,
        "type": type_,
        "statement": statement,
        "stance": stance,
        "provenance": {"quote": quote},
    }


def _entry(entry_id, title, items, topics=("rag",)) -> KBEntry:
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
            "tags": {
                "topics": list(topics),
                "knowledge_types": ["conceptual"],
                "application_forms": [],
            },
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _transcript() -> Transcript:
    return Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp="00:00:00")])


def _new_concept_response(title="Agentic RAG"):
    return json.dumps(
        [{"item_id": "k_01", "decision": "new", "title": title, "description": "d"}]
    )


def _match_response(concept_id):
    return json.dumps([{"item_id": "k_01", "decision": "match", "concept_id": concept_id}])


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


# ---- Gap 1: a concept dropped to zero members loses its OKF page too --------------------


@pytest.mark.unit
def test_gap1_sole_source_deletion_removes_orphaned_concept_page(store, embedder):
    e1 = _entry("e_01", "Video One", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    run_canonicalize_stage(e1, store, FakeClient(responses=[_new_concept_response(), "[]"]))
    concept_id = store.list_concepts()[0].concept_id
    concept_page = store.okf_root / "concepts" / f"{concept_id}.md"
    assert concept_page.exists()

    run_delete_entry_stage("e_01", store)

    assert store.load_concept(concept_id) is None
    assert not concept_page.exists()
    concepts_index = (store.okf_root / "concepts" / "index.md").read_text()
    assert concept_id not in concepts_index
    assert lint(store.okf_root) == []


# ---- Gap 2: a surviving concept's page drops the deleted video's back-reference ----------


@pytest.mark.unit
def test_gap2_surviving_concept_page_drops_stale_backreference(store, embedder):
    e1 = _entry("e_01", "Video One", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    run_canonicalize_stage(e1, store, FakeClient(responses=[_new_concept_response(), "[]"]))
    concept_id = store.list_concepts()[0].concept_id

    e2 = _entry("e_02", "Video Two", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e2, embedder=embedder, transcript=_transcript())
    run_canonicalize_stage(e2, store, FakeClient(responses=[_match_response(concept_id), "[]"]))
    assert len(store.load_concept(concept_id).members) == 2

    concept_page = store.okf_root / "concepts" / f"{concept_id}.md"
    before = concept_page.read_text()
    assert "video-one" in before

    run_delete_entry_stage("e_01", store)

    survivor = store.load_concept(concept_id)
    assert survivor is not None
    assert {m.entry_id for m in survivor.members} == {"e_02"}
    after = concept_page.read_text()
    assert "video-one" not in after
    assert "video-two" in after
    assert lint(store.okf_root) == []


# ---- Gap 3: a missing/unparseable kb file no longer strands raw/source pages -------------


@pytest.mark.unit
def test_gap3_missing_kb_file_still_strips_raw_and_source_pages(store, embedder):
    e1 = _entry("e_01", "Video One", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    source_page = store.okf_root / "sources" / "video-one.md"
    raw_page = store.okf_root / "raw" / "video-one.md"
    assert source_page.exists()
    assert raw_page.exists()

    # Simulate a missing/unparseable kb file: the entry can no longer be loaded, but its DB
    # row and OKF pages are still around, exactly like the reported production drift.
    store.entry_path("e_01").unlink()

    run_delete_entry_stage("e_01", store)

    assert not source_page.exists()
    assert not raw_page.exists()
    assert lint(store.okf_root) == []


@pytest.mark.unit
def test_gap3_unparseable_kb_file_still_strips_raw_and_source_pages(store, embedder):
    e1 = _entry("e_01", "Video One", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    source_page = store.okf_root / "sources" / "video-one.md"
    raw_page = store.okf_root / "raw" / "video-one.md"

    store.entry_path("e_01").write_text("not valid front matter at all", encoding="utf-8")

    run_delete_entry_stage("e_01", store)

    assert not source_page.exists()
    assert not raw_page.exists()
    assert lint(store.okf_root) == []
