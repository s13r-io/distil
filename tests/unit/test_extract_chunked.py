"""Chunked extraction (distil/extract.py's run_chunked_extraction) — T-EC1..EC7 (unit, FakeClient).

Covers: chunk count drives call count, item ids stay unambiguous once chunks are combined,
per-chunk truncation still surfaces on the combined result, boundary-overlap context is passed
to every chunk after the first (and only the first chunk gets none), and near-duplicate items
across chunks are folded rather than filed twice.
"""

import json

import pytest

from distil.extract import ChunkedExtractionResult, run_chunked_extraction
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import Triage

# Two short, clearly-distinct sentences. With a small chunk_chars, chunk_transcript_text (reused
# unmodified from distil/summary.py) never splits mid-sentence, so each sentence lands in its own
# chunk deterministically.
_SENT_A = "Keep functions small and focused on one job."
_SENT_B = "Write tests before writing the implementation code."


def _t(text: str) -> Transcript:
    return Transcript(segments=[Segment(text=text, locator="seg:0")])


def _triage(dominant: str = "heuristic") -> Triage:
    return Triage.model_validate(
        {
            "knowledge_types_present": [{"type": dominant, "share": 0.9}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        }
    )


def _item(statement: str, quote: str, **overrides) -> dict:
    base = {
        "type": "heuristic",
        "statement": statement,
        "stance": "opinion",
        "speaker_confidence": "high",
        "provenance": {"quote": quote, "timestamp": None, "locator": None},
    }
    base.update(overrides)
    return base


# ---- T-EC1: a transcript short enough for one chunk makes exactly one call, plain ids ----


@pytest.mark.unit
def test_single_chunk_transcript_makes_one_call_with_plain_ids():
    t = _t(_SENT_A)
    resp = json.dumps([_item("Keep functions small.", "keep functions small")])
    fake = FakeClient(responses=[resp])
    result = run_chunked_extraction(t, _triage(), fake)
    assert isinstance(result, ChunkedExtractionResult)
    assert fake.call_count == 1
    assert result.chunk_count == 1
    assert len(result.items) == 1
    # No chunk-namespacing when there's nothing to disambiguate — byte-identical to the
    # single-call run_extraction id scheme, so existing item_id-citing fixtures don't break.
    assert result.items[0].item_id == "k_01"


@pytest.mark.unit
def test_empty_transcript_makes_zero_calls():
    t = _t("")
    fake = FakeClient(responses=[])
    result = run_chunked_extraction(t, _triage(), fake)
    assert result.items == []
    assert result.chunk_count == 0
    assert fake.call_count == 0


# ---- T-EC2: multiple chunks -> one call per chunk, namespaced ids ----


@pytest.mark.unit
def test_multi_chunk_transcript_makes_one_call_per_chunk():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert result.chunk_count == 2
    assert fake.call_count == 2
    # Each chunk's own call used the same dominant type from triage.
    assert "heuristic" in fake.calls[0].prompt
    assert "heuristic" in fake.calls[1].prompt


@pytest.mark.unit
def test_multi_chunk_item_ids_are_namespaced_and_unique():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    ids = [item.item_id for item in result.items]
    assert len(ids) == len(set(ids)), "item ids must be unique once chunks are combined"
    assert ids == ["k_c00_k_01", "k_c01_k_01"]


# ---- T-EC3: boundary overlap — chunk N>0 gets trailing context from chunk N-1 ----


@pytest.mark.unit
def test_second_chunk_prompt_carries_overlap_context_from_the_first():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=30)
    first_prompt, second_prompt = fake.calls[0].prompt, fake.calls[1].prompt
    assert "[CONTEXT" not in first_prompt, "the first chunk has nothing before it to carry over"
    assert "[CONTEXT" in second_prompt
    assert "[NEW MATERIAL]" in second_prompt
    # The overlap is drawn from the tail of the previous chunk, not the new one.
    assert "focused on one job" in second_prompt.split("[NEW MATERIAL]")[0]


@pytest.mark.unit
def test_zero_overlap_chars_adds_no_context_prefix():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert "[CONTEXT" not in fake.calls[1].prompt


# ---- T-EC4: truncation on any one chunk still surfaces on the combined result ----


@pytest.mark.unit
def test_truncation_on_any_chunk_marks_the_whole_result_truncated(monkeypatch):
    import distil.extract as extract_mod

    monkeypatch.setattr(extract_mod, "_RETRY_SLEEP_SECONDS", 0)
    t = _t(f"{_SENT_A} {_SENT_B}")
    complete_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    head = _item("Write tests first.", "write tests before writing")
    truncated_b = (
        "[\n" + json.dumps(head) + ',\n  {\n    "type": "heuristic", "statement": "cut off mid'
    )
    fake = FakeClient(responses=[complete_a, truncated_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert result.truncated is True
    # The one complete item from the truncated chunk still survives.
    assert any(item.statement == "Write tests first." for item in result.items)
    assert any(item.statement == "Keep functions small." for item in result.items)


@pytest.mark.unit
def test_clean_responses_on_every_chunk_are_not_marked_truncated():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert result.truncated is False


# ---- T-EC5: near-duplicate items across chunks are folded, not filed twice ----


@pytest.mark.unit
def test_near_duplicate_items_across_chunks_are_merged():
    """The same point restated (near-identical wording) in two chunks collapses to one item."""
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps(
        [_item("Keep functions small and focused.", "keep functions small")]
    )
    resp_b = json.dumps(
        [_item("Keep functions small and focused.", "write tests before writing")]
    )
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert len(result.items) == 1


@pytest.mark.unit
def test_genuinely_different_items_across_chunks_both_survive():
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10, overlap_chars=0)
    assert len(result.items) == 2


# ---- T-EC6: chunk size / overlap are configurable, with defended defaults ----


@pytest.mark.unit
def test_default_chunk_chars_env_override(monkeypatch):
    monkeypatch.setenv("DISTIL_EXTRACT_CHUNK_CHARS", "10")
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp_a = json.dumps([_item("Keep functions small.", "keep functions small")])
    resp_b = json.dumps([_item("Write tests first.", "write tests before writing")])
    fake = FakeClient(responses=[resp_a, resp_b])
    result = run_chunked_extraction(t, _triage(), fake)
    assert result.chunk_count == 2


@pytest.mark.unit
def test_explicit_chunk_chars_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("DISTIL_EXTRACT_CHUNK_CHARS", "10")
    t = _t(f"{_SENT_A} {_SENT_B}")
    resp = json.dumps([_item("Keep functions small.", "keep functions small")])
    fake = FakeClient(responses=[resp])
    # A generous explicit chunk_chars keeps both sentences in one chunk despite the tiny env var.
    result = run_chunked_extraction(t, _triage(), fake, chunk_chars=10_000)
    assert result.chunk_count == 1
