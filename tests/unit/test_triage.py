"""Phase 3.2/3.3 — triage parsing + short-circuit. Tests T-T1, T-T2, T-T3 (unit, FakeClient)."""

import json

import pytest

from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.triage import ParseError, TriageResult, is_low_value, run_triage

_GOOD = json.dumps(
    {
        "knowledge_types_present": [
            {"type": "heuristic", "share": 0.7},
            {"type": "opinion", "share": 0.3},
        ],
        "density": "high",
        "transcript_loss": {"level": "low", "evidence": []},
        "verdict": "rich",
    }
)


# ---- T-T1: parses a well-formed model response into a TriageResult ----


@pytest.mark.unit
def test_parses_well_formed_response():
    t = ingest_text("Keep functions small. Name things clearly.")
    fake = FakeClient(responses=[_GOOD])
    result = run_triage(t, fake)
    assert isinstance(result, TriageResult)
    assert result.triage.verdict == "rich"
    assert result.triage.density == "high"
    assert result.triage.knowledge_types_present[0].type == "heuristic"
    assert fake.call_count == 1


@pytest.mark.unit
def test_tolerates_code_fence_wrapping():
    t = ingest_text("some content here")
    fenced = f"```json\n{_GOOD}\n```"
    result = run_triage(t, FakeClient(responses=[fenced]))
    assert result.triage.verdict == "rich"


# ---- T-T2: malformed/partial model JSON → clear ParseError (no silent garbage) ----


@pytest.mark.unit
def test_malformed_json_raises_parse_error():
    t = ingest_text("content")
    with pytest.raises(ParseError):
        run_triage(t, FakeClient(responses=["not json at all"]))


@pytest.mark.unit
def test_partial_json_missing_fields_raises_parse_error():
    t = ingest_text("content")
    partial = json.dumps({"density": "high"})  # missing verdict, loss, types
    with pytest.raises(ParseError):
        run_triage(t, FakeClient(responses=[partial]))


@pytest.mark.unit
def test_invalid_enum_value_raises_parse_error():
    t = ingest_text("content")
    bad = json.dumps(
        {
            "knowledge_types_present": [{"type": "gossip", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        }
    )
    with pytest.raises(ParseError):
        run_triage(t, FakeClient(responses=[bad]))


# ---- T-T3: little_to_extract short-circuit signal ----


@pytest.mark.unit
def test_is_low_value_true_for_little_to_extract():
    t = ingest_text("content")
    low = json.dumps(
        {
            "knowledge_types_present": [],
            "density": "low",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "little_to_extract",
        }
    )
    result = run_triage(t, FakeClient(responses=[low]))
    assert is_low_value(result) is True


@pytest.mark.unit
def test_is_low_value_false_for_rich():
    t = ingest_text("content")
    result = run_triage(t, FakeClient(responses=[_GOOD]))
    assert is_low_value(result) is False
