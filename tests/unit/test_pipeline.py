"""Phase 8 — pipeline.py orchestration. Tests T-PL1, T-PL2 (unit, FakeClient)."""

import json

import pytest

from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.models import KBEntry, Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.store import Store


@pytest.fixture
def profile():
    return Profile.model_validate({
        "user_id": "owner",
        "stable": {"long_term_goals": [
            {"id": "g_01", "statement": "write better code", "created_at": "2026-01-01T00:00:00"}
        ]},
        "meta": {"documents_processed": 3, "confidence": 0.3},
    })


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


_TRIAGE_RICH = json.dumps({
    "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})
_EXTRACT = json.dumps([{
    "type": "heuristic", "statement": "Keep functions small.", "stance": "opinion",
    "speaker_confidence": "high", "rationale": "easier to test", "scope": "library code",
    "provenance": {"quote": "keep functions small", "timestamp": None, "locator": None},
}])
_LINK = json.dumps([{
    "knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
    "application_form": "checklist", "scenario": "refactor auth", "novelty_flag": False,
}])
_TRIAGE_LOW = json.dumps({
    "knowledge_types_present": [], "density": "low",
    "transcript_loss": {"level": "low", "evidence": []}, "verdict": "little_to_extract",
})


# ---- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry ----


@pytest.mark.unit
def test_pl1_end_to_end_produces_valid_entry(profile, store):
    transcript = ingest_text("Keep functions small and focused on one thing.")
    # graph disabled (no prior entries anyway) → triage, extract, link = 3 calls
    client = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK])
    entry = run_pipeline(transcript, profile, store, client,
                         source_title="A talk", config=PipelineConfig(enable_graph=False))
    assert isinstance(entry, KBEntry)
    assert entry.triage.verdict == "rich"
    assert len(entry.knowledge_items) == 1
    assert entry.knowledge_items[0].provenance.quote == "keep functions small"
    assert len(entry.application_links) == 1
    assert entry.application_links[0].linked_goal_id == "g_01"
    # filed to disk + indexed
    assert store.entry_path(entry.entry_id).exists()
    assert any(r.entry_id == entry.entry_id for r in store.list_entries())


@pytest.mark.unit
def test_pl1_respects_llm_budget(profile, store):
    transcript = ingest_text("Keep functions small.")
    client = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK])
    run_pipeline(transcript, profile, store, client, source_title="t",
                 config=PipelineConfig(enable_graph=False))
    assert client.call_count <= 4  # triage + extract + link


# ---- T-PL2: little_to_extract path files a minimal entry, makes no extract/link calls ----


@pytest.mark.unit
def test_pl2_low_value_files_minimal_no_extract_calls(profile, store):
    transcript = ingest_text("hey guys smash that like button")
    client = FakeClient(responses=[_TRIAGE_LOW])  # ONLY triage; extra calls would IndexError
    entry = run_pipeline(transcript, profile, store, client, source_title="vlog")
    assert entry.triage.verdict == "little_to_extract"
    assert entry.knowledge_items == []
    assert entry.application_links == []
    # exactly one LLM call was made (triage); no extract/link/graph
    assert client.call_count == 1
    # still filed (a minimal entry)
    assert store.entry_path(entry.entry_id).exists()


@pytest.mark.unit
def test_pl_entry_id_is_unique_and_indexed(profile, store):
    t = ingest_text("Keep functions small.")
    e1 = run_pipeline(t, profile, store, FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK]),
                      source_title="t1", config=PipelineConfig(enable_graph=False))
    e2 = run_pipeline(t, profile, store, FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK]),
                      source_title="t2", config=PipelineConfig(enable_graph=False))
    assert e1.entry_id != e2.entry_id
    assert len(store.list_entries()) == 2
