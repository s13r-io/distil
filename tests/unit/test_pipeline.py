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
_NOTE = json.dumps({
    "title": "Small functions",
    "core_takeaway": {"text": "Small functions are easier to change safely.", "item_ids": ["k_01"]},
    "key_points": [{"text": "Keep the unit of behavior focused.", "item_ids": ["k_01"]}],
    "why_it_matters": [{"text": "Focused functions are easier to test.", "item_ids": ["k_01"]}],
    "how_to_apply": [{
        "text": "Use this as a refactoring checklist for auth.",
        "item_ids": ["k_01"],
        "application_link_ids": ["a_01"],
    }],
    "caveats": [{"text": "The advice is scoped to library code.", "item_ids": ["k_01"]}],
    "review_questions": [{"question": "Which function should you split first?", "item_ids": ["k_01"]}],
    "topics": ["Function Design", "Testing"],
})
_TRIAGE_LOW = json.dumps({
    "knowledge_types_present": [], "density": "low",
    "transcript_loss": {"level": "low", "evidence": []}, "verdict": "little_to_extract",
})


# ---- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry ----


@pytest.mark.unit
def test_pl1_end_to_end_produces_valid_entry(profile, store):
    transcript = ingest_text("Keep functions small and focused on one thing.")
    # graph disabled (no prior entries anyway) -> triage, extract, link, note = 4 calls
    client = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    entry = run_pipeline(transcript, profile, store, client,
                         source_title="A talk", config=PipelineConfig(enable_graph=False))
    assert isinstance(entry, KBEntry)
    assert entry.triage.verdict == "rich"
    assert len(entry.knowledge_items) == 1
    assert entry.knowledge_items[0].provenance.quote == "keep functions small"
    assert len(entry.application_links) == 1
    assert entry.application_links[0].linked_goal_id == "g_01"
    assert entry.distilled_note is not None
    assert entry.distilled_note.core_takeaway.text == "Small functions are easier to change safely."
    assert entry.tags.topics == ["function_design", "testing"]
    # filed to disk + indexed
    assert store.entry_path(entry.entry_id).exists()
    assert any(r.entry_id == entry.entry_id for r in store.list_entries())


@pytest.mark.unit
def test_pl1_respects_llm_budget(profile, store):
    transcript = ingest_text("Keep functions small.")
    client = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    run_pipeline(transcript, profile, store, client, source_title="t",
                 config=PipelineConfig(enable_graph=False))
    assert client.call_count <= 4  # triage + extract + link + note


@pytest.mark.unit
def test_pl1_reports_stage_timings(profile, store):
    transcript = ingest_text("Keep functions small.")
    client = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    timings: dict[str, float] = {}
    run_pipeline(
        transcript,
        profile,
        store,
        client,
        source_title="t",
        config=PipelineConfig(
            enable_graph=False,
            timing_callback=lambda stage, seconds: timings.__setitem__(stage, seconds),
        ),
    )
    assert {"triage", "extract", "normalize", "link", "note", "file"} <= set(timings)
    assert all(seconds >= 0 for seconds in timings.values())


# ---- T-PL2: little_to_extract path returns minimal entry, makes no extract/link calls ----


@pytest.mark.unit
def test_pl2_low_value_returns_minimal_without_filing(profile, store):
    transcript = ingest_text("hey guys smash that like button")
    client = FakeClient(responses=[_TRIAGE_LOW])  # ONLY triage; extra calls would IndexError
    entry = run_pipeline(transcript, profile, store, client, source_title="vlog")
    assert entry.triage.verdict == "little_to_extract"
    assert entry.knowledge_items == []
    assert entry.application_links == []
    # exactly one LLM call was made (triage); no extract/link/graph
    assert client.call_count == 1
    # not filed: low-value jobs remain in Activity, not the Library
    assert not store.entry_path(entry.entry_id).exists()
    assert all(row.entry_id != entry.entry_id for row in store.list_entries())


@pytest.mark.unit
def test_pl_entry_id_is_unique_and_indexed(profile, store):
    t = ingest_text("Keep functions small.")
    e1 = run_pipeline(t, profile, store,
                      FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE]),
                      source_title="t1", config=PipelineConfig(enable_graph=False))
    e2 = run_pipeline(t, profile, store,
                      FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE]),
                      source_title="t2", config=PipelineConfig(enable_graph=False))
    assert e1.entry_id != e2.entry_id
    assert len(store.list_entries()) == 2
