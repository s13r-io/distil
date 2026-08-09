"""Phase 8/15.3 — pipeline.py orchestration. Tests T-PL1, T-PL5, T-PL6 (unit, FakeClient).

There is no more quality short-circuit (owner decision — see pipeline.py's module docstring):
a transcript that reaches run_pipeline always finishes filed, regardless of triage.verdict.

Triage and extraction are merged into one strong-tier call (owner decision, addendum) — a
canned FakeClient response for that merged stage is built via `_merged(triage_json, items_json)`,
a two-section `<TRIAGE>...</TRIAGE><ITEMS>...</ITEMS>` string that replaces what used to be two
separate canned responses with one.
"""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry, Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.store import Store


def _t(text: str) -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


def _merged(triage_json: str, items_json: str) -> str:
    return f"<TRIAGE>\n{triage_json}\n</TRIAGE>\n<ITEMS>\n{items_json}\n</ITEMS>"


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
_CANON_NEW = json.dumps([{
    "item_id": "k_01", "decision": "new",
    "title": "Small Functions", "description": "Keep functions small and focused.",
}])
_SYNTH_CLAIMS = json.dumps([
    {"text": "Keep functions small and focused on one job.", "item_ids": ["k_01"]}
])
_EXTRACT_RAG = json.dumps([{
    "type": "conceptual", "statement": "Traditional RAG retrieves then generates.",
    "stance": "fact", "speaker_confidence": "high", "rationale": None, "scope": None,
    "provenance": {
        "quote": "Traditional RAG retrieves then generates", "timestamp": None, "locator": None,
    },
}])
_CANON_NEW_A = json.dumps([{
    "item_id": "k_01", "decision": "new", "title": "Concept A", "description": "d",
}])
_SYNTH_A = json.dumps([{"text": "Concept A claim.", "item_ids": ["k_01"]}])
_CANON_NEW_B = json.dumps([{
    "item_id": "k_01", "decision": "new", "title": "Concept B", "description": "d",
}])
_SYNTH_B = json.dumps([{"text": "Concept B claim.", "item_ids": ["k_01"]}])
_EDGE_RELATED = json.dumps({"relation": "related"})


# ---- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry ----


@pytest.mark.unit
def test_pl1_end_to_end_produces_valid_entry(profile, store):
    transcript = _t("Keep functions small and focused on one thing.")
    # graph disabled (no prior entries anyway) -> merged triage+extract, link, note = 3 calls
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    entry = run_pipeline(transcript, profile, store, client,
                         source_title="A talk", config=PipelineConfig(enable_graph=False, enable_canonicalize=False))
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
    transcript = _t("Keep functions small.")
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    run_pipeline(transcript, profile, store, client, source_title="t",
                 config=PipelineConfig(enable_graph=False, enable_canonicalize=False))
    assert client.call_count <= 3  # merged triage+extract + link + note


@pytest.mark.unit
def test_pl1_reports_stage_timings(profile, store):
    transcript = _t("Keep functions small.")
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    timings: dict[str, float] = {}
    run_pipeline(
        transcript,
        profile,
        store,
        client,
        source_title="t",
        config=PipelineConfig(
            enable_graph=False,
            enable_canonicalize=False,
            timing_callback=lambda stage, seconds: timings.__setitem__(stage, seconds),
        ),
    )
    assert {"extract", "normalize", "link", "note", "file"} <= set(timings)
    assert all(seconds >= 0 for seconds in timings.values())


# ---- Phase A visible progress: phase_callback reports stage entry/exit ----


@pytest.mark.unit
def test_phase_callback_reports_start_before_finish_in_stage_order(profile, store):
    """Each stage's start event must precede its own finish event, in pipeline stage order —
    and the existing timing_callback behaviour must be unaffected by adding phase_callback."""
    transcript = _t("Keep functions small.")
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    events: list[tuple[str, str]] = []
    timings: dict[str, float] = {}
    run_pipeline(
        transcript, profile, store, client, source_title="t",
        config=PipelineConfig(
            enable_graph=False,
            enable_canonicalize=False,
            timing_callback=lambda stage, seconds: timings.__setitem__(stage, seconds),
            phase_callback=lambda stage, event: events.append((stage, event)),
        ),
    )
    assert events == [
        ("extract", "start"), ("extract", "finish"),
        ("normalize", "start"), ("normalize", "finish"),
        ("link", "start"), ("link", "finish"),
        ("note", "start"), ("note", "finish"),
        ("file", "start"), ("file", "finish"),
    ]
    # timing_callback keeps working exactly as before, independent of phase_callback.
    assert {"extract", "normalize", "link", "note", "file"} <= set(timings)


@pytest.mark.unit
def test_phase_callback_never_emits_short_circuit(profile, store):
    """run_pipeline itself no longer emits a short_circuit event under any triage verdict — that
    behaviour moved to the ingest-time word-count gate, reported by callers outside this
    function (see pipeline.py's PipelineConfig.phase_callback docstring)."""
    transcript = _t("hey guys smash that like button")
    client = FakeClient(responses=[_merged(_TRIAGE_LOW, _EXTRACT), _LINK, _NOTE])
    events: list[tuple[str, str]] = []
    run_pipeline(
        transcript, profile, store, client, source_title="vlog",
        config=PipelineConfig(
            enable_graph=False, enable_canonicalize=False,
            phase_callback=lambda stage, event: events.append((stage, event)),
        ),
    )
    assert all(event != "short_circuit" for _stage, event in events)


@pytest.mark.unit
def test_phase_callback_omits_disabled_stages(profile, store):
    """graph/canonicalize/concept_edges must not appear in the events when their flags are
    off — the declared total a caller derives from these events must reflect what will
    actually run, not a fixed nine."""
    transcript = _t("Keep functions small.")
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    events: list[tuple[str, str]] = []
    run_pipeline(
        transcript, profile, store, client, source_title="t",
        config=PipelineConfig(
            enable_graph=False,
            enable_canonicalize=False,
            enable_concept_edges=False,
            phase_callback=lambda stage, event: events.append((stage, event)),
        ),
    )
    stages_seen = {stage for stage, _event in events}
    assert stages_seen == {"extract", "normalize", "link", "note", "file"}


# ---- Owner-trust: a little_to_extract verdict never stops filing (owner decision) ----


@pytest.mark.unit
def test_low_value_verdict_still_produces_a_full_filed_entry(profile, store):
    """A transcript triage would once have discarded (little_to_extract) still goes through
    extract/link/note/file exactly like any other — the owner's decision to keep the source
    overrides the model's quality judgment. Verdict is stored but informational only."""
    transcript = _t("Keep functions small and focused on one thing.")
    client = FakeClient(responses=[_merged(_TRIAGE_LOW, _EXTRACT), _LINK, _NOTE])
    entry = run_pipeline(
        transcript, profile, store, client, source_title="vlog",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
    )
    assert entry.triage.verdict == "little_to_extract"
    assert len(entry.knowledge_items) == 1
    assert entry.distilled_note is not None
    # filed to disk + indexed, exactly like a "rich" verdict
    assert store.entry_path(entry.entry_id).exists()
    assert any(row.entry_id == entry.entry_id for row in store.list_entries())


@pytest.mark.unit
def test_pl1_filing_exports_okf_pages(profile, store):
    transcript = _t("Keep functions small and focused on one thing.")
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    entry = run_pipeline(transcript, profile, store, client,
                         source_title="A talk", config=PipelineConfig(enable_graph=False, enable_canonicalize=False))
    from distil.okf import slug_for_entry

    slug = slug_for_entry(entry)
    assert (store.okf_root / "sources" / f"{slug}.md").exists()
    assert (store.okf_root / "raw" / f"{slug}.md").exists()


@pytest.mark.unit
def test_pl_entry_id_is_unique_and_indexed(profile, store):
    t = _t("Keep functions small.")
    e1 = run_pipeline(t, profile, store,
                      FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE]),
                      source_title="t1", config=PipelineConfig(enable_graph=False, enable_canonicalize=False))
    e2 = run_pipeline(t, profile, store,
                      FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE]),
                      source_title="t2", config=PipelineConfig(enable_graph=False, enable_canonicalize=False))
    assert e1.entry_id != e2.entry_id
    assert len(store.list_entries()) == 2


# ---- T-PL5: canonicalize enabled produces concept pages under the okf_root ----


@pytest.mark.unit
def test_pl5_canonicalize_enabled_produces_concept_pages(profile, store):
    transcript = _t("Keep functions small and focused on one thing.")
    client = FakeClient(
        responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE, _CANON_NEW, _SYNTH_CLAIMS]
    )
    run_pipeline(
        transcript, profile, store, client, source_title="A talk",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True),
    )
    concepts_dir = store.okf_root / "concepts"
    pages = [p for p in concepts_dir.glob("*.md") if p.name != "index.md"]
    assert len(pages) == 1
    text = pages[0].read_text()
    assert "type: concept" in text
    assert "## Sources" in text


# ---- T-PL6: enable_canonicalize=False makes zero canonicalize-stage LLM calls -----------


@pytest.mark.unit
def test_pl6_canonicalize_disabled_makes_zero_canonicalize_calls(profile, store):
    transcript = _t("Keep functions small and focused on one thing.")
    # Only the 3 core-stage responses; a canonicalize call would IndexError.
    client = FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT), _LINK, _NOTE])
    run_pipeline(
        transcript, profile, store, client, source_title="A talk",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=False),
    )
    assert client.call_count == 3  # merged triage+extract + link + note; zero canonicalize calls
    assert store.list_concepts() == []
    concepts_dir = store.okf_root / "concepts"
    pages = [p for p in concepts_dir.glob("*.md") if p.name != "index.md"]
    assert pages == []


# ---- Phase 16 — Stage 9 concept<->concept typed edges (design report §9 item 4) --------------


@pytest.mark.unit
def test_pl7_concept_edges_enabled_computes_and_renders_edges(profile, store):
    embedder = FakeEmbedder(dim=32)
    transcript = _t("Traditional RAG retrieves then generates.")

    run_pipeline(
        transcript, profile, store,
        FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT_RAG), _LINK, _NOTE, _CANON_NEW_A, _SYNTH_A]),
        source_title="Video A",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True, enable_concept_edges=True),
        embedder=embedder,
    )
    run_pipeline(
        transcript, profile, store,
        FakeClient(
            responses=[
                _merged(_TRIAGE_RICH, _EXTRACT_RAG), _LINK, _NOTE, _CANON_NEW_B, _SYNTH_B, _EDGE_RELATED,
            ]
        ),
        source_title="Video B",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True, enable_concept_edges=True),
        embedder=embedder,
    )

    concepts = store.list_concepts()
    assert len(concepts) == 2
    concept_b = next(c for c in concepts if c.title == "Concept B")
    assert concept_b.edges != []
    page = store.okf_root / "concepts" / f"{concept_b.concept_id}.md"
    assert "## Related" in page.read_text()


@pytest.mark.unit
def test_pl8_concept_edges_disabled_makes_zero_edge_calls(profile, store):
    embedder = FakeEmbedder(dim=32)
    transcript = _t("Traditional RAG retrieves then generates.")

    run_pipeline(
        transcript, profile, store,
        FakeClient(responses=[_merged(_TRIAGE_RICH, _EXTRACT_RAG), _LINK, _NOTE, _CANON_NEW_A, _SYNTH_A]),
        source_title="Video A",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True, enable_concept_edges=False),
        embedder=embedder,
    )
    # No _EDGE_RELATED response supplied; an edge-classification call would IndexError.
    client = FakeClient(
        responses=[_merged(_TRIAGE_RICH, _EXTRACT_RAG), _LINK, _NOTE, _CANON_NEW_B, _SYNTH_B]
    )
    run_pipeline(
        transcript, profile, store, client,
        source_title="Video B",
        config=PipelineConfig(enable_graph=False, enable_canonicalize=True, enable_concept_edges=False),
        embedder=embedder,
    )

    assert client.call_count == 5
    concept_b = next(c for c in store.list_concepts() if c.title == "Concept B")
    assert concept_b.edges == []
