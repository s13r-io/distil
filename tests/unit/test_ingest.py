"""Phase 1.3 — ingest.py (stage 0, PURE). Tests T-I1..I6.

Normalizes .srt/.txt/.md/pasted text into one transcript: a list of segments
{text, timestamp?, locator}. Timestamps are captured when the source has them and left
null otherwise; a locator is always populated.

Also covers the ingest-time word-count gate (owner decision — the pipeline's only quality
rule): a transcript below the minimum word count is rejected with TranscriptTooShortError,
distinct from IngestError's other failure modes (missing file, empty input, bad format).
"""

import re
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


# ---- Inline MM:SS rolling-caption dumps (YouTube transcript-panel paste shape) -----------
#
# Fixture is a verbatim 100-line excerpt of a real YouTube transcript-panel export (see
# CLAUDE.md's ingest entry) — never hand-typed — so these tests can't share a wrong assumption
# with the implementation the way a hand-typed fixture once did for a different bug.


@pytest.mark.unit
def test_mmss_rolling_caption_fixture_parses_clean_speech_and_timestamps():
    t = ingest_file(FIX / "youtube_rolling_caption_mmss.txt")
    assert len(t.segments) == 34
    assert t.segments[0].text == "Um, hi everyone. Good evening. I think"
    assert t.segments[0].timestamp == "00:00:58"
    assert t.segments[1].timestamp == "00:01:00"
    # timestamps normalized to HH:MM:SS, consistent with the other inline-timestamp path
    assert all(s.timestamp is not None and s.timestamp.count(":") == 2 for s in t.segments)


@pytest.mark.unit
def test_mmss_rolling_caption_word_count_reflects_real_prose_not_timestamp_tokens():
    """The old paragraph fallback spliced every MM:SS token into the text as if spoken;
    the real defect this fixes: on this excerpt that would have counted ~34 extra
    timestamp-shaped tokens as words. The fixed word count must be prose-only."""
    t = ingest_file(FIX / "youtube_rolling_caption_mmss.txt")
    assert len(t.full_text().split()) == 196


@pytest.mark.unit
def test_mmss_rolling_caption_leaks_no_timestamp_tokens_into_text():
    t = ingest_file(FIX / "youtube_rolling_caption_mmss.txt")
    for seg in t.segments:
        assert not re.search(r"\b\d{1,2}:\d{2}\b", seg.text), seg.text


@pytest.mark.unit
def test_mmss_rolling_caption_preview_lines_do_not_duplicate_or_appear_as_segments():
    """Rolling playback previews the next timestamp on its own text-less line just before
    the real text line for it arrives — that preview must be dropped, not kept as an empty
    segment and not merged into the following segment's text."""
    t = ingest_file(FIX / "youtube_rolling_caption_mmss.txt")
    assert all(seg.text.strip() for seg in t.segments)
    # "I'm audible to all of you guys. Let me" / "know in the chat..." are two distinct
    # segments in the source, not one merged/duplicated line.
    texts = [seg.text for seg in t.segments]
    assert texts.count("I'm audible to all of you guys. Let me") == 1
    assert texts.count("know in the chat if I'm audible.") == 1


@pytest.mark.unit
def test_ordinary_prose_without_timestamps_is_unaffected_by_mmss_detection():
    prose = (
        "We covered a lot of ground today, from architecture to deployment.\n\n"
        f"{_LONG_ENOUGH} and then some more discussion followed after that point."
    )
    t = ingest_text(prose)
    assert len(t.segments) == 2
    assert all(s.timestamp is None for s in t.segments)


@pytest.mark.unit
def test_a_single_spoken_time_is_not_mistaken_for_a_caption_timestamp():
    """A speaker legitimately saying something that looks like a timestamp must not flip
    ordinary prose into caption parsing — only one line out of many opens with a digit
    pattern here, well under the majority bar, so it stays plain text untouched."""
    prose = (
        "So here's the plan for the demo today.\n"
        "3:15 is when we'll actually start the live portion of the walkthrough.\n"
        f"{_LONG_ENOUGH}\n"
        "After that we'll take questions from the audience for a while.\n"
        "Thanks everyone for showing up early to help us test the setup.\n"
    )
    t = ingest_text(prose)
    assert all(s.timestamp is None for s in t.segments)
    assert any("3:15 is when we'll actually start" in s.text for s in t.segments)


@pytest.mark.unit
def test_mmss_shaped_but_non_monotonic_timestamps_refuse_clearly():
    """Every line opens with an MM:SS-shaped token (clears the majority bar), but the
    timestamps jump backwards — not real caption output, so this must refuse rather than
    silently mis-parse. Mangled and correct must never look identical."""
    bogus = "\n".join(
        [
            f"01:00 {_LONG_ENOUGH}",
            "00:30 second line goes backwards in time which real captions never do",
            "02:00 third line",
        ]
    )
    with pytest.raises(IngestError):
        ingest_text(bogus)


@pytest.mark.unit
def test_mmss_shaped_but_partially_unmatched_lines_refuse_clearly():
    """Majority of lines look like MM:SS captions but one real line doesn't match at all —
    an ambiguous, mixed shape this must refuse rather than guess about."""
    bogus = "\n".join(
        [
            f"01:00 {_LONG_ENOUGH}",
            "02:00 second line",
            "this line has no timestamp at all and breaks the caption shape",
            "03:00 fourth line",
        ]
    )
    with pytest.raises(IngestError):
        ingest_text(bogus)


@pytest.mark.unit
def test_srt_path_is_unaffected_by_mmss_caption_detection():
    """The live YouTube/collector path (ingest_srt_text -> _parse_srt) puts timestamps in
    the segment's timestamp field already and never routes through ingest_text's MM:SS
    detection at all — confirm its output is exactly what it was before this fix."""
    t = ingest_file(FIX / "sample.srt")
    assert len(t.segments) == 3
    assert t.segments[0].timestamp == "00:00:01"
    assert t.segments[1].timestamp == "00:00:04"
    assert t.segments[2].timestamp == "00:01:12"
    assert "Welcome to the talk" in t.segments[0].text


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
