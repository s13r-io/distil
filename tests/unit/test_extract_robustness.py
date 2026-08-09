"""Extraction robustness: bounded retry + truncated-array tolerance. T-E5, T-E6, T-E7.

Covers the live-service failure `Extraction response was not a JSON array: '```json\\n[\\n {\\n
"type": "conceptual", ...'` — the model's response was truncated mid-array (4,096-token output
cap, or a dropped connection) and the old strict `json.loads` rejected the whole thing, throwing
away items that had already fully parsed.
"""

import json
import logging

import pytest

import distil.extract as extract
from distil.extract import _parse_items_json, run_extraction, run_triage_extract
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import Triage
from distil.triage import ParseError


def _t(text: str) -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


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


_TRIAGE_JSON = json.dumps({
    "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})


def _merged_complete(items_json_array: str) -> str:
    return f"<TRIAGE>\n{_TRIAGE_JSON}\n</TRIAGE>\n<ITEMS>\n{items_json_array}\n</ITEMS>"


def _merged_truncated(head_obj: dict) -> str:
    """A merged triage+extract response whose <ITEMS> array is cut off mid-object and never
    closed — mirrors _truncated_array's shape (one complete leading item, then a second item
    cut off mid-string) but wrapped in the two-section format run_triage_extract parses."""
    items_text = (
        "[\n"
        + json.dumps(head_obj)
        + ',\n  {\n    "type": "conceptual",\n    "statement": "Second item cut off mid'
    )
    return f"<TRIAGE>\n{_TRIAGE_JSON}\n</TRIAGE>\n<ITEMS>\n{items_text}"


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
    t = _t("some transcript text about a concept")
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
    t = _t(overlong_quote + " and more transcript text")
    raw = _truncated_array(head)
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[raw]))
    assert len(items) == 1
    assert len(items[0].provenance.quote.split()) <= 14


# ---- T-E15: a salvaged/truncated response must never look identical to a complete one ----


@pytest.mark.unit
def test_parse_items_json_reports_truncated_true_only_when_salvaged():
    data, truncated = _parse_items_json(_truncated_array(_COMPLETE_ITEM), kind="Extraction")
    assert truncated is True
    assert len(data) == 1


@pytest.mark.unit
def test_parse_items_json_reports_truncated_false_for_a_clean_response():
    data, truncated = _parse_items_json(_VALID_SINGLE_ITEM_RESPONSE, kind="Extraction")
    assert truncated is False
    assert len(data) == 1


@pytest.mark.unit
def test_parse_items_json_reports_truncated_false_when_only_fence_stripping_was_needed():
    """A complete array wrapped in a code fence (or surrounding prose) needed only cosmetic
    stripping, not salvage — it must not be confused with a genuinely truncated response."""
    fenced = "Sure, here you go:\n```json\n" + _VALID_SINGLE_ITEM_RESPONSE + "\n```"
    data, truncated = _parse_items_json(fenced, kind="Extraction")
    assert truncated is False
    assert len(data) == 1


@pytest.mark.unit
def test_run_triage_extract_reports_truncated_when_salvaged():
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[_merged_truncated(_COMPLETE_ITEM)])
    result = run_triage_extract(t, fake)
    assert result.truncated is True
    assert len(result.items) == 1
    assert result.items[0].statement == "First complete item."


@pytest.mark.unit
def test_run_triage_extract_complete_response_is_not_marked_truncated():
    """A clean, complete response must never be falsely flagged as truncated — a salvage that
    silently looks identical to success is the defect this flag exists to prevent, and the
    inverse (crying wolf on a normal response) would be just as unacceptable."""
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[_merged_complete(json.dumps([_COMPLETE_ITEM]))])
    result = run_triage_extract(t, fake)
    assert result.truncated is False
    assert len(result.items) == 1


@pytest.mark.unit
def test_run_triage_extract_logs_a_clear_warning_when_truncated(caplog):
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[_merged_truncated(_COMPLETE_ITEM)])
    with caplog.at_level(logging.WARNING, logger="distil.extract"):
        run_triage_extract(t, fake)
    assert any(
        "truncated" in record.getMessage().lower() and record.levelno == logging.WARNING
        for record in caplog.records
    )


@pytest.mark.unit
def test_run_triage_extract_does_not_log_a_truncation_warning_when_complete(caplog):
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[_merged_complete(json.dumps([_COMPLETE_ITEM]))])
    with caplog.at_level(logging.WARNING, logger="distil.extract"):
        run_triage_extract(t, fake)
    assert not any("truncated" in record.getMessage().lower() for record in caplog.records)


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
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[ConnectionError("dropped"), _VALID_SINGLE_ITEM_RESPONSE])
    items = run_extraction(t, _triage("conceptual"), fake)
    assert len(items) == 1
    assert fake.call_count == 2


@pytest.mark.unit
def test_persistent_parse_failure_raises_parse_error_after_bounded_retries():
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=["not json at all"] * 3)  # 1 + _MAX_RETRIES attempts
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 3


@pytest.mark.unit
def test_persistent_connection_failure_raises_after_bounded_retries():
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[ConnectionError("dropped")] * 3)
    with pytest.raises(ConnectionError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 3


@pytest.mark.unit
def test_semantic_schema_failure_is_not_retried():
    """A complete, parseable array whose item fails schema validation is a semantic failure —
    it must raise immediately, with no retry (bullet 1)."""
    bad_item = {"type": "conceptual", "statement": "missing stance and provenance"}
    t = _t("some transcript text about a concept")
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
    t = _t("someone shares a personal story about debugging")
    resp = json.dumps([_item(type="personal_experience", stance="personal_experience")])
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert items[0].type == "conceptual"
    assert items[0].stance == "personal_experience"


@pytest.mark.unit
def test_valid_but_non_requested_type_is_left_alone():
    """A `type` that IS a valid KnowledgeType, just not the requested one, is not overwritten —
    the model is allowed to flag an item as a genuinely different type."""
    t = _t("a mix of concept and opinion content")
    resp = json.dumps([_item(type="opinion", stance="opinion")])
    items = run_extraction(t, _triage("conceptual"), FakeClient(responses=[resp]))
    assert items[0].type == "opinion"


@pytest.mark.unit
def test_unrecoverable_item_is_dropped_while_valid_siblings_survive():
    """One item that fails validation even after the type repair (missing required fields) is
    dropped, not fatal — its valid siblings in the same batch still come back."""
    bad_item = {"type": "personal_experience", "statement": "missing stance and provenance"}
    resp = json.dumps([_item(), _item(statement="Second complete item."), bad_item])
    t = _t("some transcript text about a concept")
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
    t = _t("some transcript text about a concept")
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
    t = _t("some transcript text about a concept")
    fake = FakeClient(responses=[resp])
    with pytest.raises(ParseError):
        run_extraction(t, _triage("conceptual"), fake)
    assert fake.call_count == 1
