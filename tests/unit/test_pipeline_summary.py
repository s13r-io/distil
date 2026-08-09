"""Narrative summary stage wired into pipeline.py — additive, cheap-tier, opt-in via
``summary_client``. Proves the model-tier split by counting calls on two distinct FakeClient
instances (one per tier), never by inspecting configuration."""

import json

import pytest

from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.models import Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.store import Store

_TRIAGE_RICH = json.dumps({
    "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})
_EXTRACT = json.dumps([{
    "type": "heuristic", "statement": "Keep functions small.", "stance": "opinion",
    "speaker_confidence": "high",
    "provenance": {"quote": "keep functions small", "timestamp": None, "locator": None},
}])
_LINK = json.dumps([])
_NOTE = json.dumps({
    "title": "Small functions",
    "core_takeaway": {"text": "Small functions are easier to reason about.", "item_ids": ["k_01"]},
    "key_points": [], "why_it_matters": [], "how_to_apply": [], "caveats": [],
    "review_questions": [], "topics": ["functions"],
})

# Repeats the extracted item's quote so normalize.py's faithfulness gate keeps the item (and
# link/note stages actually run) — 300 repeats is well within the default chunk size, so the
# narrative-summary stage makes exactly one call (chunk summary only, no merge).
_TRANSCRIPT_TEXT = "Keep functions small. " * 300


@pytest.fixture
def profile():
    return Profile.model_validate({"user_id": "owner"})


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


@pytest.mark.unit
def test_narrative_summary_runs_on_a_separate_cheap_client(profile, store):
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    narrative_text = "N" * 600
    cheap = FakeClient(responses=[narrative_text])

    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )

    assert entry.narrative_summary is not None
    assert entry.narrative_summary.text == narrative_text
    # Strong client did exactly triage+extract+link+note — zero narrative-summary calls landed
    # on it, and the cheap client did exactly the one narrative-summary call — zero
    # triage/extract/link/note calls landed on it. This is the call-count proof, not a
    # configuration inspection.
    assert strong.call_count == 4
    assert cheap.call_count == 1


@pytest.mark.unit
def test_narrative_summary_skipped_without_a_summary_client(profile, store):
    """Every caller that existed before this stage was added omits summary_client — behavior
    must reproduce exactly: no extra call, entry.narrative_summary stays None."""
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
    )
    assert entry.narrative_summary is None
    assert strong.call_count == 4


@pytest.mark.unit
def test_narrative_summary_respects_its_own_enable_flag(profile, store):
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=[])  # would IndexError if ever called
    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(
            enable_graph=False, enable_canonicalize=False, enable_narrative_summary=False,
        ),
        summary_client=cheap,
    )
    assert entry.narrative_summary is None
    assert cheap.call_count == 0


@pytest.mark.unit
def test_narrative_summary_failure_does_not_block_filing(profile, store):
    """A cheap client that only ever returns too-thin output exhausts its retries and fails
    internally — the pipeline must still file the entry, just without a narrative summary."""
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=["x", "x", "x"])  # always too short; 3 == default max_retries

    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )

    assert entry.narrative_summary is None
    assert store.entry_path(entry.entry_id).exists()
    assert cheap.call_count == 3


@pytest.mark.unit
def test_narrative_summary_dropped_connection_does_not_block_filing(profile, store):
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=[ConnectionError("dropped")])

    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )

    assert entry.narrative_summary is None
    assert store.entry_path(entry.entry_id).exists()


@pytest.mark.unit
def test_narrative_summary_phase_events_land_between_note_and_graph(profile, store):
    transcript = ingest_text(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=["N" * 600])
    events: list[tuple[str, str]] = []
    run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(
            enable_graph=False, enable_canonicalize=False,
            phase_callback=lambda stage, event: events.append((stage, event)),
        ),
        summary_client=cheap,
    )
    assert events == [
        ("triage", "start"), ("triage", "finish"),
        ("extract", "start"), ("extract", "finish"),
        ("normalize", "start"), ("normalize", "finish"),
        ("link", "start"), ("link", "finish"),
        ("note", "start"), ("note", "finish"),
        ("narrative_summary", "start"), ("narrative_summary", "finish"),
        ("file", "start"), ("file", "finish"),
    ]
