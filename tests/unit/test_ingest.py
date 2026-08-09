"""Phase 1.3 — ingest.py (stage 0, PURE). Tests T-I1..I6.

Normalizes .srt/.txt/.md/pasted text into one transcript: a list of segments
{text, timestamp?, locator}. Timestamps are captured when the source has them and left
null otherwise; a locator is always populated.

Also covers the ingest-time word-count gate (owner decision — the pipeline's only quality
rule): a transcript below the minimum word count is rejected with TranscriptTooShortError,
distinct from IngestError's other failure modes (missing file, empty input, bad format).
"""

from pathlib import Path

import pytest

from distil.ingest import (
    IngestError,
    Transcript,
    TranscriptTooShortError,
    ingest_file,
    ingest_srt_text,
    ingest_text,
    is_thin_source,
)

FIX = Path(__file__).parent.parent / "fixtures"

_LONG_ENOUGH = " ".join(["word"] * 60)


# ---- T-I1: parse sample.srt → ordered segments with parsed timestamps ----


@pytest.mark.unit
def test_srt_parses_to_ordered_timestamped_segments():
    t = ingest_file(FIX / "sample.srt")
    assert isinstance(t, Transcript)
    assert len(t.segments) == 3
    assert t.segments[0].text.startswith("Welcome to the talk")
    assert t.segments[0].timestamp == "00:00:01"
    assert t.segments[1].timestamp == "00:00:04"
    assert t.segments[2].timestamp == "00:01:12"
    # ordered
    assert [s.locator for s in t.segments] == ["seg:0", "seg:1", "seg:2"]


# ---- T-I2: inline HH:MM:SS markers captured ----


@pytest.mark.unit
def test_inline_timestamps_captured():
    t = ingest_file(FIX / "inline_ts.txt")
    assert len(t.segments) == 3
    assert t.segments[0].timestamp == "00:00:05"
    assert t.segments[2].timestamp == "00:12:30"
    # the marker is stripped from the text
    assert "00:00:05" not in t.segments[0].text
    assert "testing strategy" in t.segments[0].text


# ---- T-I3: no timestamps → null timestamp + populated locator ----


@pytest.mark.unit
def test_no_timestamps_yields_null_ts_and_locator():
    t = ingest_file(FIX / "no_timestamps.md")
    assert len(t.segments) >= 3
    assert all(s.timestamp is None for s in t.segments)
    assert all(s.locator for s in t.segments)
    # markdown heading is not treated as a knowledge segment
    assert all(not s.text.startswith("#") for s in t.segments)


# ---- T-I4: pasted plain text normalized same as .txt ----


@pytest.mark.unit
def test_pasted_text_normalized_like_txt():
    pasted = f"First line of pasted notes. {_LONG_ENOUGH}\n\nSecond paragraph here."
    t = ingest_text(pasted)
    assert len(t.segments) == 2
    assert t.segments[0].timestamp is None
    assert t.segments[0].locator == "seg:0"


# ---- T-I5: unknown/binary file or empty input → clear error, not a crash ----


@pytest.mark.unit
def test_empty_input_raises_ingest_error():
    with pytest.raises(IngestError):
        ingest_text("   \n  \n")


@pytest.mark.unit
def test_unknown_extension_raises_ingest_error(tmp_path):
    p = tmp_path / "video.mp4"
    p.write_bytes(b"\x00\x01\x02binarygarbage")
    with pytest.raises(IngestError):
        ingest_file(p)


@pytest.mark.unit
def test_missing_file_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError):
        ingest_file(tmp_path / "does_not_exist.txt")


# ---- T-I6: normalized shape identical across formats (downstream is format-agnostic) ----


@pytest.mark.unit
def test_uniform_shape_across_formats():
    srt = ingest_file(FIX / "sample.srt")
    txt = ingest_file(FIX / "inline_ts.txt")
    md = ingest_file(FIX / "no_timestamps.md")
    paste = ingest_text(f"a {_LONG_ENOUGH}\n\nb")
    for t in (srt, txt, md, paste):
        assert isinstance(t, Transcript)
        for seg in t.segments:
            # every segment has these three attributes; timestamp may be None
            assert hasattr(seg, "text") and seg.text
            assert hasattr(seg, "timestamp")
            assert hasattr(seg, "locator") and seg.locator


@pytest.mark.unit
def test_full_text_helper_joins_segments():
    t = ingest_text(f"alpha {_LONG_ENOUGH}\n\nbeta")
    assert "alpha" in t.full_text() and "beta" in t.full_text()


# ---- Owner decision: word-count gate is the pipeline's only quality rule -----------------


@pytest.mark.unit
def test_transcript_below_minimum_words_is_rejected_on_every_ingest_path():
    """Applies uniformly: pasted text, uploaded .txt/.md, uploaded .srt, and fetched/collected
    captions (ingest_srt_text) — every public entry point bottoms out in the same two
    low-level parsers (_parse_paragraphs/_parse_inline_timestamped via ingest_text, and
    _parse_srt via ingest_file's .srt branch and ingest_srt_text)."""
    short_text = "Only a handful of words here."
    with pytest.raises(TranscriptTooShortError):
        ingest_text(short_text)

    short_srt = "1\n00:00:01,000 --> 00:00:02,000\nToo short.\n"
    with pytest.raises(TranscriptTooShortError):
        ingest_srt_text(short_srt)

    with pytest.raises(TranscriptTooShortError):
        ingest_text(f"00:00:01 {short_text}")  # inline-timestamped branch too


@pytest.mark.unit
def test_transcript_too_short_error_is_an_ingest_error_but_distinguishable():
    """A subclass of IngestError (it is an ingest-time rejection) but callers can and must
    branch on the specific type to tell "too short" apart from a genuine read/parse failure."""
    assert issubclass(TranscriptTooShortError, IngestError)
    with pytest.raises(IngestError):
        ingest_text("too short")


@pytest.mark.unit
def test_transcript_at_minimum_word_count_is_accepted():
    exactly_50 = " ".join(["word"] * 50)
    t = ingest_text(exactly_50)
    assert len(t.full_text().split()) == 50


@pytest.mark.unit
def test_transcript_one_word_under_minimum_is_rejected():
    forty_nine = " ".join(["word"] * 49)
    with pytest.raises(TranscriptTooShortError):
        ingest_text(forty_nine)


@pytest.mark.unit
def test_min_word_count_is_configurable(monkeypatch):
    monkeypatch.setenv("DISTIL_MIN_TRANSCRIPT_WORDS", "5")
    # 6 words clears a floor of 5, though it would fail the default 50.
    t = ingest_text("one two three four five six")
    assert len(t.full_text().split()) == 6

    monkeypatch.setenv("DISTIL_MIN_TRANSCRIPT_WORDS", "100")
    with pytest.raises(TranscriptTooShortError):
        ingest_text(_LONG_ENOUGH)  # 60 words clears the default but not a floor of 100


@pytest.mark.unit
def test_rejection_message_states_word_count_and_minimum():
    with pytest.raises(TranscriptTooShortError, match=r"3 words.*minimum 50"):
        ingest_text("only three words")


@pytest.mark.unit
def test_a_transcript_well_above_minimum_is_never_rejected():
    """A long, low-quality transcript (what triage would once have called
    little_to_extract) is not a second rejection rule — only the word count gates."""
    rubbish = " ".join(["like", "um", "you", "know"] * 20)  # 80 low-content words
    t = ingest_text(rubbish)
    assert len(t.segments) >= 1


# ---- is_thin_source: a visibility signal, never a rejection ------------------------------


@pytest.mark.unit
def test_is_thin_source_true_below_the_thin_threshold():
    assert is_thin_source(200) is True


@pytest.mark.unit
def test_is_thin_source_false_at_or_above_the_thin_threshold():
    assert is_thin_source(500) is False
    assert is_thin_source(1000) is False


@pytest.mark.unit
def test_is_thin_source_false_for_zero_word_count():
    """Zero means "unknown" (an entry filed before this field existed), not "extremely
    thin" — must not be flagged, to avoid mislabeling old data."""
    assert is_thin_source(0) is False


@pytest.mark.unit
def test_thin_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("DISTIL_THIN_TRANSCRIPT_WORDS", "50")
    assert is_thin_source(60) is False
    assert is_thin_source(40) is True
