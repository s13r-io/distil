"""Extraction robustness: bounded retry + truncated-array tolerance. T-E5, T-E6, T-E7.

Covers the live-service failure `Extraction response was not a JSON array: '```json\\n[\\n {\\n
"type": "conceptual", ...'` — the model's response was truncated mid-array (4,096-token output
cap, or a dropped connection) and the old strict `json.loads` rejected the whole thing, throwing
away items that had already fully parsed.
"""

import json

import pytest

import distil.extract as extract
from distil.extract import _parse_items_json, run_extraction
from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.models import Triage
from distil.triage import ParseError


def _triage(dominant: str) -> Triage:
    return Triage.model_validate(
        {
            "knowledge_types_present": [{"type": dominant, "share": 0.9}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        }
    )


_COMPLETE_ITEM = {
    "type": "conceptual",
    "statement": "First complete item.",
    "stance": "fact",
    "speaker_confidence": "high",
    "provenance": {"quote": "first complete item", "timestamp": None, "locator": None},
}

_VALID_SINGLE_ITEM_RESPONSE = json.dumps([_COMPLETE_ITEM])


def _truncated_array(head_obj: dict) -> str:
    """A JSON array (with a leading code fence, matching real model output) containing one
    complete item followed by a second item cut off mid-string — as if the output-token cap
    or a dropped connection truncated the stream after the first item."""
    return (
        "```json\n[\n"
        + json.dumps(head_obj)
        + ',\n  {\n    "type": "conceptual",\n    "statement": "Second item cut off mid'
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(extract, "_RETRY_SLEEP_SECONDS", 0)


# ---- T-E5: truncated array whose complete prefix parses -> recover the complete items ----


@pytest.mark.unit
def test_truncated_array_recovers_complete_leading_items():
    t = ingest_text("some transcript text about a concept")
    raw = _truncated_array(_COMPLETE_ITEM)
    fake = FakeClient(responses=[raw])
    items = run_extraction(t, _triage("conceptual"), fake)
    assert len(items) == 1
    assert items[0].statement == "First complete item."
    assert fake.call_count == 1  # a successful recovery is not a retry-worthy failure


@pytest.mark.unit
def test_truncated_array_recovery_still_applies_quote_truncation_and_discipline():
    """Bullet 3: recovered items still go through _truncate_overlong_quotes /
    _enforce_quote_discipline, same as normally-parsed items."""
    overlong_quote = " ".join(["word"] * 20)
    head = {**_COMPLETE_ITEM, "provenance": {"quote": overlong_quote, "timestamp": None, "locator": None}}
    t = ingest_text(overlong_quote + " and more transcript text")
    raw = _truncated_array(head)
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[raw]))
    assert len(items) == 1
    assert len(items[0].provenance.quote.split()) <= 14


# ---- T-E6: first object itself incomplete -> nothing recoverable, clean ParseError ----


@pytest.mark.unit
def test_first_object_incomplete_raises_parse_error_cleanly():
    """When even the first array element is cut off, recovery yields nothing useful. This
    must reject cleanly (ParseError) rather than crash with some unrelated exception, and
    must never fabricate a partial object."""
    raw = '```json\n[\n  {\n    "type": "conceptual",\n    "statement": "cut off mid'
    with pytest.raises(ParseError):
        _parse_items_json(raw, kind="Extraction")


@pytest.mark.unit
def test_non_array_response_still_rejected():
    with pytest.raises(ParseError):
        _parse_items_json("Sure, here is my analysis: nothing to extract.", kind="Extraction")


# ---- T-E7: dropped connection retried, and persistent failure raises ----


@pytest.mark.unit
def test_dropped_connection_retries_then_succeeds():
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=[ConnectionError("dropped"), _VALID_SINGLE_ITEM_RESPONSE])
    items = run_extraction(t, _triage("conceptual"), fake)
    assert len(items) == 1
    assert fake.call_count == 2


@pytest.mark.unit
def test_persistent_parse_failure_raises_parse_error_after_bounded_retries():
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=["not json at all"] * 3)  # 1 + _MAX_RETRIES attempts
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 3


@pytest.mark.unit
def test_persistent_connection_failure_raises_after_bounded_retries():
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=[ConnectionError("dropped")] * 3)
    with pytest.raises(ConnectionError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 3


@pytest.mark.unit
def test_semantic_schema_failure_is_not_retried():
    """A complete, parseable array whose item fails schema validation is a semantic failure —
    it must raise immediately, with no retry (bullet 1)."""
    bad_item = {"type": "conceptual", "statement": "missing stance and provenance"}
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=[json.dumps([bad_item])])
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 1
