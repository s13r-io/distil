"""Narrative summary stage wired into pipeline.py — additive, cheap-tier, opt-in via
``summary_client``, and running CONCURRENTLY with triage+extract (owner decision, addendum
Part 2 — supersedes an earlier "runs after note" placement). Proves the model-tier split by
counting calls on two distinct FakeClient instances (one per tier), never by inspecting
configuration."""

import json
import time

import pytest

from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.store import Store


@pytest.fixture(autouse=True)
def _disable_unslop_for_summary_concurrency_contracts(monkeypatch):
    """Keep this file's client counts focused on narrative-summary scheduling and retries."""
    monkeypatch.setenv("DISTIL_UNSLOP_ENABLED", "false")

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


def _t(text: str) -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


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
    transcript = _t(_TRANSCRIPT_TEXT)
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
    # Strong client did exactly one triage call + one extract call, plus link+note — zero
    # narrative-summary calls landed on it, and the cheap client did exactly the one
    # narrative-summary call — zero triage/extract/link/note calls landed on it. This is the
    # call-count proof, not a configuration inspection.
    assert strong.call_count == 4
    assert cheap.call_count == 1


@pytest.mark.unit
def test_narrative_summary_skipped_without_a_summary_client(profile, store):
    """Every caller that existed before this stage was added omits summary_client — behavior
    must reproduce exactly: no extra call, entry.narrative_summary stays None."""
    transcript = _t(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
    )
    assert entry.narrative_summary is None
    assert strong.call_count == 4


@pytest.mark.unit
def test_narrative_summary_respects_its_own_enable_flag(profile, store):
    transcript = _t(_TRANSCRIPT_TEXT)
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
    transcript = _t(_TRANSCRIPT_TEXT)
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
    transcript = _t(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=[ConnectionError("dropped")])

    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )

    assert entry.narrative_summary is None
    assert store.entry_path(entry.entry_id).exists()


# ---- Addendum Part 2: concurrent with triage+extract, started at the front -----------------


@pytest.mark.unit
def test_narrative_summary_phase_events_start_at_front_and_finish_after_note(profile, store):
    """narrative_summary's "start" fires before triage's "start" (it kicks off the front of
    the pipeline); its "finish" fires only once joined, after note — everything else remains a
    strictly sequential start/finish stream from the phase_callback's point of view, since it is
    only ever invoked from the main thread."""
    transcript = _t(_TRANSCRIPT_TEXT)
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
        ("narrative_summary", "start"),
        ("triage", "start"), ("triage", "finish"),
        ("extract", "start"), ("extract", "finish"),
        ("normalize", "start"), ("normalize", "finish"),
        ("link", "start"), ("link", "finish"),
        ("note", "start"), ("note", "finish"),
        ("narrative_summary", "finish"),
        ("file", "start"), ("file", "finish"),
    ]


class _DelayedClient:
    """A minimal LLMClient that sleeps before returning a canned response — used to prove two
    stages actually overlap in wall-clock time rather than merely being wired to run "in
    parallel" in name only."""

    def __init__(self, responses: list, delay_seconds: float):
        self._responses = list(responses)
        self._delay = delay_seconds
        self._cursor = 0
        self.calls: list[float] = []  # perf_counter() at call time

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(time.perf_counter())
        time.sleep(self._delay)
        response = self._responses[self._cursor]
        self._cursor += 1
        if isinstance(response, BaseException):
            raise response
        return response

    def stream(self, prompt: str, *, system: str | None = None):
        yield self.complete(prompt, system=system)


@pytest.mark.unit
def test_narrative_summary_actually_overlaps_extraction_in_wall_clock_time(profile, store):
    """The real proof of concurrency: both the strong (triage+extract+link+note) and cheap
    (summary) clients sleep for a fixed duration per call. If they ran sequentially, the total
    wall-clock time would be at least (strong calls + cheap calls) * delay; running concurrently,
    it's close to just the strong side's total (the longer of the two), since the cheap side
    starts at the same time and finishes well before note does."""
    transcript = _t(_TRANSCRIPT_TEXT)
    delay = 0.2
    strong = _DelayedClient([_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE], delay_seconds=delay)
    cheap = _DelayedClient(["N" * 600], delay_seconds=delay)

    start = time.perf_counter()
    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )
    elapsed = time.perf_counter() - start

    assert entry.narrative_summary is not None
    # Sequential would take >= 5 * delay (4 strong calls + 1 cheap call); concurrent should stay
    # well under 4 * delay plus generous scheduling slack.
    assert elapsed < delay * 4.5, f"elapsed {elapsed:.2f}s looks sequential, not concurrent"
    # The cheap call started at essentially the same time as the first strong call — proof the
    # summary genuinely starts at the front rather than being merely scheduled "eventually".
    assert abs(cheap.calls[0] - strong.calls[0]) < delay


@pytest.mark.unit
def test_narrative_summary_timeout_leaves_a_clear_not_generated_state(monkeypatch, profile, store, caplog):
    """A summary that loses the race (still running past the bound) must never delay filing
    indefinitely, and its absence must be logged honestly rather than silent."""
    monkeypatch.setenv("DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS", "0.05")
    transcript = _t(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = _DelayedClient(["N" * 600], delay_seconds=1.0)  # far longer than the 0.05s bound

    import logging
    with caplog.at_level(logging.WARNING, logger="distil.pipeline"):
        entry = run_pipeline(
            transcript, profile, store, strong, source_title="t",
            config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
            summary_client=cheap,
        )

    assert entry.narrative_summary is None
    assert store.entry_path(entry.entry_id).exists()  # still files, never blocked
    assert any("still running" in record.message for record in caplog.records)


@pytest.mark.unit
def test_extraction_failure_is_not_masked_by_a_concurrently_succeeding_summary(profile, store):
    """An extraction failure must still surface as a real exception — a summary succeeding on
    its own background thread must never hide it."""
    transcript = _t(_TRANSCRIPT_TEXT)
    # No valid <TRIAGE>/<ITEMS> response queued at all for either retry attempt.
    strong = FakeClient(responses=["not a valid response", "still not valid", "nope"])
    cheap = FakeClient(responses=["N" * 600])

    from distil.triage import ParseError
    with pytest.raises(ParseError):
        run_pipeline(
            transcript, profile, store, strong, source_title="t",
            config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
            summary_client=cheap,
        )


@pytest.mark.unit
def test_summary_failure_is_not_masked_by_extraction_succeeding(profile, store):
    """The reverse: a failing summary must be reported (as "no summary", logged) independently
    of extraction's own success — extraction succeeding must not swallow the summary's outcome."""
    transcript = _t(_TRANSCRIPT_TEXT)
    strong = FakeClient(responses=[_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    cheap = FakeClient(responses=[RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])

    entry = run_pipeline(
        transcript, profile, store, strong, source_title="t",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
        summary_client=cheap,
    )
    assert len(entry.knowledge_items) == 1  # extraction succeeded fully
    assert entry.narrative_summary is None  # summary's own failure, independently reported
