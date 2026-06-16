"""Phase 4.1/4.2 — type-routed extraction + quote discipline. T-E1, T-E2, T-E4 (unit)."""

import json

import pytest

from distil.extract import QuoteDisciplineError, run_extraction
from distil.ingest import ingest_text
from distil.llm import FakeClient
from distil.models import Triage
from distil.prompts.extract import build_extract_prompt


def _triage(dominant: str) -> Triage:
    return Triage.model_validate(
        {
            "knowledge_types_present": [{"type": dominant, "share": 0.9}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        }
    )


_HEURISTIC_RESPONSE = json.dumps(
    [
        {
            "type": "heuristic",
            "statement": "Keep functions focused on a single responsibility.",
            "stance": "opinion",
            "speaker_confidence": "high",
            "rationale": "Easier to name and test.",
            "scope": "When writing library code.",
            "provenance": {"quote": "keep functions small", "timestamp": None, "locator": None},
        }
    ]
)

_PROCEDURAL_RESPONSE = json.dumps(
    [
        {
            "type": "procedural",
            "statement": "Create a virtual environment first.",
            "stance": "fact",
            "speaker_confidence": "high",
            "order_index": 0,
            "preconditions": [],
            "gotchas": ["Installing before creating pollutes global site-packages."],
            "provenance": {"quote": "create a new virtual environment", "timestamp": None, "locator": None},
        }
    ]
)


# ---- T-E1: routes to the right extractor based on triage dominant type ----


@pytest.mark.unit
def test_routes_to_heuristic_extractor():
    t = ingest_text("keep functions small and focused on one thing")
    fake = FakeClient(responses=[_HEURISTIC_RESPONSE])
    items = run_extraction(t, _triage("heuristic"), fake)
    assert len(items) == 1
    assert items[0].type == "heuristic"
    # the prompt that was sent must be the heuristic-routed one
    assert "heuristic" in fake.calls[0].prompt


@pytest.mark.unit
def test_routes_to_procedural_extractor():
    t = ingest_text("first create a new virtual environment, then install")
    fake = FakeClient(responses=[_PROCEDURAL_RESPONSE])
    items = run_extraction(t, _triage("procedural"), fake)
    assert items[0].type == "procedural"
    assert "procedural" in fake.calls[0].prompt


# ---- T-E2: type-specific fields present ----


@pytest.mark.unit
def test_heuristic_items_have_rationale_and_scope():
    t = ingest_text("keep functions small and focused on one thing")
    items = run_extraction(t, _triage("heuristic"), FakeClient(responses=[_HEURISTIC_RESPONSE]))
    assert items[0].rationale
    assert items[0].scope


@pytest.mark.unit
def test_procedural_items_have_order_index():
    t = ingest_text("first create a new virtual environment, then install")
    items = run_extraction(t, _triage("procedural"), FakeClient(responses=[_PROCEDURAL_RESPONSE]))
    assert items[0].order_index == 0


# ---- T-E4: quote discipline < 15 words enforced in code ----


@pytest.mark.unit
def test_overlong_quote_is_truncated():
    """A verbatim quote that is slightly over the word limit is truncated, not rejected.

    The LLM sometimes returns a genuine verbatim quote that is one or two words over
    the 15-word ceiling.  Truncating to 14 words preserves faithfulness (a leading
    substring is still verbatim) while satisfying the copyright guardrail.
    """
    long_quote = " ".join(["word"] * 20)
    t = ingest_text(long_quote + " and more text here")
    resp = json.dumps(
        [
            {
                "type": "heuristic",
                "statement": "Something.",
                "stance": "opinion",
                "speaker_confidence": "medium",
                "provenance": {"quote": long_quote, "timestamp": None, "locator": None},
            }
        ]
    )
    items = run_extraction(t, _triage("heuristic"), FakeClient(responses=[resp]))
    assert len(items) == 1
    assert len(items[0].provenance.quote.split()) <= 14, "truncation must keep quote under limit"


@pytest.mark.unit
def test_overlong_quote_guard_raises_directly():
    """_enforce_quote_discipline still raises when called directly with an over-limit quote.

    This preserves the T-E4 code-level guarantee:  the guard itself is strict.  The
    truncation step in run_extraction is what prevents genuine verbatim LLM quotes from
    reaching the guard as over-limit strings.
    """
    from distil.models import KnowledgeItem
    item = KnowledgeItem.model_validate({
        "item_id": "k_01",
        "type": "heuristic",
        "statement": "Something.",
        "stance": "opinion",
        "speaker_confidence": "medium",
        "provenance": {"quote": " ".join(["word"] * 20), "timestamp": None, "locator": None},
    })
    from distil.extract import _enforce_quote_discipline
    with pytest.raises(QuoteDisciplineError):
        _enforce_quote_discipline([item])


@pytest.mark.unit
def test_quote_exactly_14_words_allowed():
    quote = " ".join([f"w{i}" for i in range(14)])
    t = ingest_text(quote + " trailing")
    resp = json.dumps(
        [
            {
                "type": "opinion",
                "statement": "Ok.",
                "stance": "opinion",
                "speaker_confidence": "medium",
                "provenance": {"quote": quote, "timestamp": None, "locator": None},
            }
        ]
    )
    items = run_extraction(t, _triage("opinion"), FakeClient(responses=[resp]))
    assert len(items) == 1


@pytest.mark.unit
def test_item_ids_are_assigned():
    t = ingest_text("keep functions small")
    items = run_extraction(t, _triage("heuristic"), FakeClient(responses=[_HEURISTIC_RESPONSE]))
    assert items[0].item_id


@pytest.mark.unit
def test_prompt_builder_includes_type_specific_fields():
    assert "rationale" in build_extract_prompt("heuristic", "x")
    assert "order_index" in build_extract_prompt("procedural", "x")
