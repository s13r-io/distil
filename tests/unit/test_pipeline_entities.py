"""Phase D — entities flow end-to-end through run_pipeline: extracted alongside knowledge items
in the same extract call, canonicalized/synthesized in the same Stage 8 as concepts, exported to
the OKF bundle. No new pipeline stage, no extra transcript read.
"""

import json

import pytest

from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import Profile
from distil.pipeline import PipelineConfig, run_pipeline
from distil.store import Store


def _t(text: str) -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


_TRIAGE_RICH = json.dumps({
    "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})
_EXTRACT_WITH_ENTITY = json.dumps([{
    "type": "conceptual", "statement": "React renders UI using a virtual DOM.",
    "stance": "fact", "speaker_confidence": "high",
    "provenance": {"quote": "react renders ui using a virtual dom", "timestamp": None, "locator": None},
    "entities": [{
        "name": "React", "kind": "tool", "description": "A JS UI library.",
        "quote": "react renders ui using a virtual dom", "timestamp": None,
    }],
}])
_LINK = json.dumps([])
_NOTE = json.dumps({
    "title": "React basics",
    "core_takeaway": {"text": "React uses a virtual DOM.", "item_ids": ["k_01"]},
    "topics": ["frontend"],
})
_CANON_NEW_CONCEPT = json.dumps([{
    "item_id": "k_01", "decision": "new", "title": "Virtual DOM rendering", "description": "d",
}])
_SYNTH_CONCEPT_CLAIMS = json.dumps([{"text": "Concept claim.", "item_ids": ["k_01"]}])
_CANON_NEW_ENTITY = json.dumps([{
    "mention_key": "k_01#0", "decision": "new", "title": "React", "description": "A JS UI library.",
}])
_SYNTH_ENTITY_CLAIMS = json.dumps([{"text": "React uses a virtual DOM.", "item_ids": ["k_01"]}])


@pytest.fixture
def profile():
    return Profile(user_id="owner")


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


@pytest.mark.unit
def test_entity_flows_end_to_end_through_the_pipeline_no_extra_transcript_read(profile, store):
    transcript = _t("react renders ui using a virtual dom under the hood")
    client = FakeClient(responses=[
        _TRIAGE_RICH, _EXTRACT_WITH_ENTITY, _LINK, _NOTE,
        _CANON_NEW_CONCEPT, _SYNTH_CONCEPT_CLAIMS,
        _CANON_NEW_ENTITY, _SYNTH_ENTITY_CLAIMS,
    ])
    entry = run_pipeline(
        transcript, profile, store, client, source_title="Intro to React",
        config=PipelineConfig(enable_graph=False, enable_concept_edges=False),
    )

    # Exactly 8 model calls were made — nothing extra, no second transcript read (the single
    # extract call above already carries the entity; every later call operates on
    # already-extracted data, never the transcript itself).
    assert client.call_count == 8
    assert len(entry.knowledge_items[0].entity_mentions) == 1

    entities = store.list_entities()
    assert len(entities) == 1
    assert entities[0].title == "React"
    assert entities[0].kind == "tool"
    assert len(entities[0].members) == 1

    entity_page = store.okf_root / "entities" / f"{entities[0].entity_id}.md"
    assert entity_page.exists()
    text = entity_page.read_text()
    assert "type: entity" in text
    assert "kind: tool" in text


@pytest.mark.unit
def test_enable_entities_false_makes_zero_entity_llm_calls(profile, store):
    transcript = _t("react renders ui using a virtual dom under the hood")
    # Only 6 responses: triage/extract/link/note/concept-canon/concept-synth. An entity
    # call would IndexError on the FakeClient if enable_entities weren't honored.
    client = FakeClient(responses=[
        _TRIAGE_RICH, _EXTRACT_WITH_ENTITY, _LINK, _NOTE,
        _CANON_NEW_CONCEPT, _SYNTH_CONCEPT_CLAIMS,
    ])
    run_pipeline(
        transcript, profile, store, client, source_title="Intro to React",
        config=PipelineConfig(enable_graph=False, enable_concept_edges=False, enable_entities=False),
    )
    assert store.list_entities() == []
