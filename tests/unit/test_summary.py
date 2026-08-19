"""Narrative summary layer (distil/summary.py) — chunking, coverage-floor retry/give-up, and
chronological merge. All model calls go through FakeClient; no real model calls."""

import pytest

from distil.llm import FakeClient
from distil.summary import (
    NarrativeSummaryError,
    chunk_transcript_text,
    synthesize_narrative_summary,
)

# ---- chunking -----------------------------------------------------------------------------

_SENTENCES = [
    "Sentence one is fairly short.",
    "Sentence two is also fairly short indeed.",
    "Sentence three adds a good deal more content here to lengthen things.",
    "Sentence four.",
    "Sentence five wraps everything up rather nicely at the very end.",
]


@pytest.mark.unit
def test_chunking_keeps_sentences_whole_and_respects_target_size():
    text = " ".join(_SENTENCES)
    chunks = chunk_transcript_text(text, chunk_chars=50)

    assert len(chunks) > 1
    # No sentence dropped, duplicated, reordered, or split across a chunk boundary.
    assert " ".join(chunks) == text
    for sentence in _SENTENCES:
        containing = [c for c in chunks if sentence in c]
        assert len(containing) == 1

    for chunk in chunks:
        # Either the chunk respects the target, or it's a single sentence too long to split
        # further (the "keep sentences whole" guarantee wins over the size target).
        assert len(chunk) <= 50 or chunk in _SENTENCES


@pytest.mark.unit
def test_chunking_keeps_a_single_oversized_sentence_whole():
    long_sentence = "This one sentence alone is much longer than the target chunk size here."
    text = f"Short lead-in. {long_sentence} Short trailer."
    chunks = chunk_transcript_text(text, chunk_chars=20)
    assert any(chunk == long_sentence for chunk in chunks)


@pytest.mark.unit
def test_chunking_env_default_is_configurable(monkeypatch):
    text = " ".join(_SENTENCES)
    monkeypatch.setenv("DISTIL_SUMMARY_CHUNK_CHARS", "30")
    small = chunk_transcript_text(text)
    monkeypatch.setenv("DISTIL_SUMMARY_CHUNK_CHARS", "10000")
    large = chunk_transcript_text(text)
    assert len(small) > len(large) == 1


@pytest.mark.unit
def test_chunking_empty_input_yields_no_chunks():
    assert chunk_transcript_text("   ") == []


# ---- coverage-floor retry / give-up ---------------------------------------------------------


@pytest.mark.unit
def test_thin_chunk_summary_is_rejected_and_retried():
    text = "word " * 500  # 2500 chars -> min chunk summary length = 200 chars
    good = "A" * 250
    client = FakeClient(responses=["too short", good])
    result = synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert result.text == good
    assert client.call_count == 2  # one rejected attempt, one accepted


@pytest.mark.unit
def test_dropped_connection_is_retried_like_a_thin_result():
    text = "word " * 500
    good = "A" * 250
    client = FakeClient(responses=[ConnectionError("dropped"), good])
    result = synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert result.text == good
    assert client.call_count == 2


@pytest.mark.unit
def test_thin_chunk_summary_gives_up_honestly_after_bound():
    text = "word " * 500
    client = FakeClient(responses=["short"] * 5)  # more than max_retries; would IndexError if used
    with pytest.raises(NarrativeSummaryError):
        synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert client.call_count == 3


@pytest.mark.unit
def test_thin_merge_is_also_rejected_and_retried():
    sentences = [
        "First idea explained in some real detail right here and now for good measure.",
        "Second idea explained afterward in similar detail too for good measure as well.",
    ]
    text = " ".join(sentences)
    chunk_summary_1 = "B" * 80
    chunk_summary_2 = "C" * 80
    thin_merge = "D" * 10
    good_merge = "E" * 100
    client = FakeClient(
        responses=[chunk_summary_1, chunk_summary_2, thin_merge, good_merge]
    )
    result = synthesize_narrative_summary(text, client, chunk_chars=20, max_retries=3)
    assert result.text == good_merge
    assert client.call_count == 4


# ---- merge preserves chronological order -----------------------------------------------------


@pytest.mark.unit
def test_merge_preserves_chronological_order():
    sentences = [
        "First idea explained in some real detail right here and now for good measure.",
        "Second idea explained afterward in similar detail too for good measure as well.",
    ]
    text = " ".join(sentences)
    chunk_summary_1 = "B" * 60
    chunk_summary_2 = "C" * 60
    merged = "D" * 60
    client = FakeClient(responses=[chunk_summary_1, chunk_summary_2, merged])

    result = synthesize_narrative_summary(text, client, chunk_chars=20, max_retries=1)

    assert result.chunk_count == 2
    assert result.text == merged
    assert client.call_count == 3
    merge_prompt = client.calls[-1].prompt
    assert merge_prompt.index(chunk_summary_1) < merge_prompt.index(chunk_summary_2)


@pytest.mark.unit
def test_single_chunk_skips_the_merge_call():
    text = "word " * 500
    good = "A" * 250
    client = FakeClient(responses=[good])
    result = synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert result.chunk_count == 1
    assert client.call_count == 1  # no merge call needed for a single chunk


@pytest.mark.unit
def test_unslop_retries_when_rewrite_falls_below_merged_coverage_floor():
    text = "word " * 500
    original = "A" * 250
    thin = "too short"
    polished = "B" * 220
    client = FakeClient([original, thin, thin, polished, polished])

    result = synthesize_narrative_summary(
        text, client, chunk_chars=10_000, max_retries=2, unslop_client=client
    )

    assert result.text == polished
    assert client.call_count == 5


# ---- cheap-model tagging -----------------------------------------------------------------


@pytest.mark.unit
def test_result_model_tag_comes_from_the_injected_client():
    text = "word " * 500
    client = FakeClient(responses=["A" * 250])
    client.model = "claude-haiku-4-5"  # mirrors AnthropicClient's public `.model` attribute
    result = synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert result.model == "claude-haiku-4-5"


@pytest.mark.unit
def test_result_model_tag_falls_back_to_the_summary_stage_resolver(monkeypatch):
    """A client with no .model attribute (FakeClient's default) still tags the result with
    the "summary" stage's own resolved model, not a hardcoded literal."""
    monkeypatch.setenv("DISTIL_MODEL_SUMMARY", "claude-opus-5")
    text = "word " * 500
    client = FakeClient(responses=["A" * 250])
    result = synthesize_narrative_summary(text, client, chunk_chars=10_000, max_retries=3)
    assert result.model == "claude-opus-5"


@pytest.mark.unit
def test_empty_transcript_raises_honestly():
    with pytest.raises(NarrativeSummaryError):
        synthesize_narrative_summary("   ", FakeClient(responses=[]))
