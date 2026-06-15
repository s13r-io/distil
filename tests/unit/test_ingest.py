"""Phase 1.3 — ingest.py (stage 0, PURE). Tests T-I1..I6.

Normalizes .srt/.txt/.md/pasted text into one transcript: a list of segments
{text, timestamp?, locator}. Timestamps are captured when the source has them and left
null otherwise; a locator is always populated.
"""

from pathlib import Path

import pytest

from distil.ingest import IngestError, Transcript, ingest_file, ingest_text

FIX = Path(__file__).parent.parent / "fixtures"


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
    pasted = "First line of pasted notes.\n\nSecond paragraph here."
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
    paste = ingest_text("a\n\nb")
    for t in (srt, txt, md, paste):
        assert isinstance(t, Transcript)
        for seg in t.segments:
            # every segment has these three attributes; timestamp may be None
            assert hasattr(seg, "text") and seg.text
            assert hasattr(seg, "timestamp")
            assert hasattr(seg, "locator") and seg.locator


@pytest.mark.unit
def test_full_text_helper_joins_segments():
    t = ingest_text("alpha\n\nbeta")
    assert "alpha" in t.full_text() and "beta" in t.full_text()
