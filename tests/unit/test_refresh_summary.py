"""Per-entry narrative-summary refresh (distil/refresh_summary.py) — generate/regenerate ONLY
the narrative summary from the stored raw transcript; never re-fetch, never re-extract."""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry, Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.refresh_summary import refresh_narrative_summary
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
_CANON_NEW = json.dumps([{
    "item_id": "k_01", "decision": "new", "title": "Small Functions",
    "description": "Keep functions small and focused.",
}])
_SYNTH_CLAIMS = json.dumps([{"text": "Keep functions small and focused.", "item_ids": ["k_01"]}])


def _merged(triage_json: str, items_json: str) -> str:
    return f"<TRIAGE>\n{triage_json}\n</TRIAGE>\n<ITEMS>\n{items_json}\n</ITEMS>"


@pytest.fixture
def profile():
    return Profile.model_validate({"user_id": "owner"})


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


def _file_real_entry(profile, store):
    """A fully filed entry with a concept membership, via the real pipeline — so refresh's
    "leaves concepts/entities/items untouched" claim has something real to verify against."""
    transcript = Transcript(segments=[Segment(
        text="Keep functions small and focused on one job. It makes testing dramatically easier.",
        locator="seg:0",
    )])
    client = FakeClient(
        responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE, _CANON_NEW, _SYNTH_CLAIMS]
    )
    embedder = FakeEmbedder(dim=16)
    return run_pipeline(
        transcript, profile, store, client, source_title="A talk",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True),
        embedder=embedder,
    )


@pytest.mark.unit
def test_refresh_generates_summary_and_leaves_items_concepts_untouched(profile, store):
    entry = _file_real_entry(profile, store)
    concepts_before = store.list_concepts()
    assert len(concepts_before) == 1
    items_before = store.load_entry(entry.entry_id).knowledge_items
    page_path = store.okf_root / "concepts" / f"{concepts_before[0].concept_id}.md"
    page_before = page_path.read_text()
    raw_page_path = next((store.okf_root / "raw").glob("*.md"))
    raw_page_before = raw_page_path.read_text()

    summary_text = "N" * 200
    result = refresh_narrative_summary(
        entry.entry_id, store, FakeClient(responses=[summary_text])
    )

    assert result.ok is True
    reloaded = store.load_entry(entry.entry_id)
    assert reloaded.narrative_summary is not None
    assert reloaded.narrative_summary.text == summary_text
    assert reloaded.knowledge_items == items_before

    assert store.list_concepts() == concepts_before
    assert page_path.read_text() == page_before
    assert raw_page_path.read_text() == raw_page_before


@pytest.mark.unit
def test_refresh_calls_only_the_narrative_summary_stage(profile, store):
    """A shared client with exactly one canned response would IndexError if refresh made any
    call beyond the single chunk summary — e.g. a re-run of extraction, note, or canonicalize."""
    entry = _file_real_entry(profile, store)
    client = FakeClient(responses=["N" * 200])
    result = refresh_narrative_summary(entry.entry_id, store, client)
    assert result.ok is True
    assert client.call_count == 1


@pytest.mark.unit
def test_refresh_reports_missing_transcript_clearly(store):
    entry = KBEntry.model_validate({
        "entry_id": "e_no_transcript",
        "source": {"title": "Old entry", "captured_at": "2026-01-01T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01", "type": "heuristic", "statement": "x", "stance": "opinion",
            "provenance": {"quote": "x"},
        }],
        "meta": {"created_at": "2026-01-01T00:00:00", "model_version": "test"},
    })
    store.file_entry(entry)  # no transcript= kwarg -> no raw/ page is ever written

    result = refresh_narrative_summary("e_no_transcript", store, FakeClient(responses=[]))
    assert result.ok is False
    assert "transcript" in result.message.lower()
    assert "fetch" in result.message.lower()  # explains why retrying alone won't help


@pytest.mark.unit
def test_refresh_reports_unknown_entry(store):
    result = refresh_narrative_summary("e_missing", store, FakeClient(responses=[]))
    assert result.ok is False
    assert "not found" in result.message.lower()


@pytest.mark.unit
def test_refresh_gives_up_honestly_on_thin_summary(profile, store):
    entry = _file_real_entry(profile, store)
    client = FakeClient(responses=["x", "x", "x"])  # always too short; 3 == default max_retries
    result = refresh_narrative_summary(entry.entry_id, store, client)
    assert result.ok is False
    assert "could not generate" in result.message.lower()
    assert store.load_entry(entry.entry_id).narrative_summary is None
