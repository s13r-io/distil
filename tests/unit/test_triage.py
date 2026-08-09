"""Phase 3.2/3.3 — triage parsing. Tests T-T1, T-T2 (unit, FakeClient).

Triage no longer gates the pipeline (owner decision — see distil/pipeline.py's module
docstring); it only classifies. There is no is_low_value/short-circuit to test here anymore."""

import json

import pytest

from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.triage import ParseError, TriageResult, run_triage


def _t(text: str = "Keep functions small. Name things clearly.") -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


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
    t = _t()
    fake = FakeClient(responses=[_GOOD])
    result = run_triage(t, fake)
    assert isinstance(result, TriageResult)
    assert result.triage.verdict == "rich"
    assert result.triage.density == "high"
    assert result.triage.knowledge_types_present[0].type == "heuristic"
    assert fake.call_count == 1


@pytest.mark.unit
def test_tolerates_code_fence_wrapping():
    t = _t("some content here")
    fenced = f"```json\n{_GOOD}\n```"
    result = run_triage(t, FakeClient(responses=[fenced]))
    assert result.triage.verdict == "rich"


# ---- T-T2: malformed/partial model JSON → clear ParseError (no silent garbage) ----


@pytest.mark.unit
def test_malformed_json_raises_parse_error():
    t = _t("content")
    with pytest.raises(ParseError):
        run_triage(t, FakeClient(responses=["not json at all"]))


@pytest.mark.unit
def test_partial_json_missing_fields_raises_parse_error():
    t = _t("content")
    partial = json.dumps({"density": "high"})  # missing verdict, loss, types
    with pytest.raises(ParseError):
        run_triage(t, FakeClient(responses=[partial]))


@pytest.mark.unit
def test_invalid_enum_value_raises_parse_error():
    t = _t("content")
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
