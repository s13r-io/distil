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
    head = {
        **_COMPLETE_ITEM,
        "provenance": {"quote": overlong_quote, "timestamp": None, "locator": None},
    }
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


# ---- T-E8: type/stance drift (reported live-service failure) ----


def _item(**overrides) -> dict:
    base = {
        "type": "conceptual",
        "statement": "A complete item.",
        "stance": "fact",
        "speaker_confidence": "high",
        "provenance": {"quote": "a complete item", "timestamp": None, "locator": None},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_stance_value_in_type_field_is_repaired_to_requested_type():
    """Reproduces the reported error: 'Extracted item 18 did not match the schema: 1
    validation error for KnowledgeItem type ... input_value=\\'personal_experience\\''.

    The model copied a `stance` value into `type`. Since the caller always knows the exact
    `KnowledgeType` it asked for (`build_extract_prompt`), this is repaired rather than
    dropped or fatal.
    """
    t = ingest_text("someone shares a personal story about debugging")
    resp = json.dumps([_item(type="personal_experience", stance="personal_experience")])
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert items[0].type == "conceptual"
    assert items[0].stance == "personal_experience"


@pytest.mark.unit
def test_valid_but_non_requested_type_is_left_alone():
    """A `type` that IS a valid KnowledgeType, just not the requested one, is not overwritten —
    the model is allowed to flag an item as a genuinely different type."""
    t = ingest_text("a mix of concept and opinion content")
    resp = json.dumps([_item(type="opinion", stance="opinion")])
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[resp]))
    assert items[0].type == "opinion"


@pytest.mark.unit
def test_unrecoverable_item_is_dropped_while_valid_siblings_survive():
    """One item that fails validation even after the type repair (missing required fields) is
    dropped, not fatal — its valid siblings in the same batch still come back."""
    bad_item = {"type": "personal_experience", "statement": "missing stance and provenance"}
    resp = json.dumps([_item(), _item(statement="Second complete item."), bad_item])
    t = ingest_text("some transcript text about a concept")
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[resp]))
    assert len(items) == 2
    assert {i.statement for i in items} == {"A complete item.", "Second complete item."}


@pytest.mark.unit
def test_all_bad_array_still_raises():
    """When every item is unrecoverable, that's a wholesale-broken response — raise rather than
    silently returning an empty list."""
    bad_items = [
        {"type": "personal_experience", "statement": "missing stance and provenance"},
        {"type": "personal_experience", "statement": "also missing stance and provenance"},
    ]
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=[json.dumps(bad_items)])
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 1  # schema-level failure — still not retried


@pytest.mark.unit
def test_below_salvage_floor_raises_even_though_some_items_survive():
    """A response where most items are broken (well under the 50% salvage floor) is treated as
    systemically broken and raises, rather than silently returning the one surviving item."""
    bad_item = {"type": "personal_experience", "statement": "missing stance and provenance"}
    resp = json.dumps([_item(), bad_item, bad_item, bad_item])
    t = ingest_text("some transcript text about a concept")
    fake = FakeClient(responses=[resp])
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 1
