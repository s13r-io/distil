"""Stage 0 — Ingest (PURE, no LLM). ARCHITECTURE.md §2; PRD FR21/FR22; TESTING T-I*.

Parses any supported input — ``.srt``, ``.txt``, ``.md``, or pasted text — into one
normalized :class:`Transcript`: an ordered list of :class:`Segment` ``{text, timestamp?,
locator}``. Timestamps are captured when the source has them (SRT cues, inline ``HH:MM:SS``
markers, or inline ``MM:SS`` rolling-caption dumps — see below) and left ``None`` otherwise; a
``locator`` (``seg:<index>``) is always populated so untimestamped sources still have a stable
pointer. Downstream stages depend only on this shape, never on the original format.

**Inline ``MM:SS`` rolling-caption dumps (fixes a real defect: pasted YouTube transcripts were
silently mangled).** Copying YouTube's transcript panel produces a shape the old inline-``HH:MM:SS``
detector never recognized: every non-blank line opens with a bare ``MM:SS`` timestamp, and rolling
playback duplicates each upcoming timestamp on its own text-less line just before the real text
line for it arrives (confirmed against a real ~4,600-line export). Treating that as ordinary prose
(the old fallback) spliced every timestamp digit into the middle of the speech — on the owner's own
~8,600-word file this injected 3,073 timestamp tokens (26% word-count inflation) and broke verbatim
quote matching badly enough that extraction dropped every item. ``_parse_mmss_rolling_caption``
handles it: text-less lines are discarded as caption-window noise, real lines become segments with
their ``MM:SS`` converted to ``HH:MM:SS`` for display, and text stays untouched. Detecting this shape
is deliberately conservative given how easily ``MM:SS`` collides with spoken content (`"at 3:15 we
start"`): a majority of non-blank lines must open with the pattern (mirroring the existing
``HH:MM:SS`` bar) *and*, inside the parser itself, EVERY non-blank line must match and the matched
timestamps must be non-decreasing start-to-end — genuine caption exports satisfy both; scattered
prose mentioning clock times essentially never does. Either check failing raises ``IngestError``
rather than guessing which lines are captions.

**The only quality gate in the pipeline is a word count (owner decision, supersedes the old
triage ``little_to_extract`` short-circuit).** ``_check_min_words`` runs at the tail of
``_parse_srt`` and ``ingest_text`` — the two low-level parsers every public entry point
(``ingest_file``, ``ingest_text``, ``ingest_srt_text``) bottoms out in — so every ingest path
(pasted text, uploaded file, server-fetched YouTube video, external-collector submission) is
covered by these two call sites without needing one of its own. It raises
:class:`TranscriptTooShortError`, a distinct subclass of :class:`IngestError`, so callers can
tell "too short to work with" apart from a genuine read/parse/fetch failure. The owner has been
explicit that this is the *only* rejection rule: no duration, no coverage arithmetic, no model
judgment of quality.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_INLINE_TS_HHMMSS = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2})(?:[.,]\d+)?\s+(.*)$")
_INLINE_TS_MMSS = re.compile(r"^\s*(\d{1,2}:\d{2})[ \t]*(.*)$")
_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,]\d{3}\s*-->")
_SRT_INDEX_ONLY = re.compile(r"^\d+$")

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".vtt"}

# Owner decision: "if the transcript has less than 50 words, it should be rejected. More than
# that, you should work on it." Configurable via DISTIL_MIN_TRANSCRIPT_WORDS for the same reason
# every other tuned threshold in this codebase is env-overridable (see store.py's
# DISTIL_CONCEPT_SIM_FLOOR etc.) — the default is the owner's stated number.
MIN_TRANSCRIPT_WORDS = 50

# A separate, higher, non-rejecting threshold: below this, a filed entry is flagged as built from
# unusually little material (see is_thin_source below). Never gates — see its docstring.
THIN_TRANSCRIPT_WORDS = 500


class IngestError(ValueError):
    """Raised for empty input, a missing file, or an unsupported/binary format."""


class TranscriptTooShortError(IngestError):
    """Raised when a transcript has fewer than the minimum word count worth distilling.

    A subclass of :class:`IngestError` (it *is* an ingest-time rejection) but distinct enough
    for callers to branch on: this is a clear, expected rejection of thin content, never a
    fetch/parse/read failure, and must be presented to the owner differently from one."""


def _min_transcript_words() -> int:
    try:
        return int(os.environ.get("DISTIL_MIN_TRANSCRIPT_WORDS", MIN_TRANSCRIPT_WORDS))
    except ValueError:
        return MIN_TRANSCRIPT_WORDS


def _check_min_words(transcript: Transcript) -> None:
    word_count = len(transcript.full_text().split())
    minimum = _min_transcript_words()
    if word_count < minimum:
        raise TranscriptTooShortError(
            f"Transcript has only {word_count} word{'s' if word_count != 1 else ''} "
            f"(minimum {minimum}) — too short to work with."
        )


def is_thin_source(word_count: int) -> bool:
    """True when a filed entry's transcript is unusually short for its source.

    Deliberately transcript-only, and deliberately just a visibility signal, never a rejection:
    a partially truncated fetch (e.g. an hour-long video whose fetch died after three minutes)
    can't be told apart from a genuinely short source without knowing the source's real
    duration, and ``Source.duration_sec`` is never populated today (see models.py/pipeline.py) —
    populating it would mean touching the transcript-fetching machinery, which this gate must
    not do. So this only ever flags "built from little material," honestly, without claiming to
    detect truncation specifically. ``word_count == 0`` means unknown (entries filed before this
    field existed) and is deliberately not flagged, to avoid mislabeling old data as thin.
    """
    return 0 < word_count < _thin_transcript_words()


def _thin_transcript_words() -> int:
    try:
        return int(os.environ.get("DISTIL_THIN_TRANSCRIPT_WORDS", THIN_TRANSCRIPT_WORDS))
    except ValueError:
        return THIN_TRANSCRIPT_WORDS


@dataclass
class Segment:
    text: str
    locator: str
    timestamp: str | None = None


@dataclass
class Transcript:
    segments: list[Segment]

    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)


def ingest_file(path: str | Path) -> Transcript:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".srt":
        raw = _read_text(p)
        return _parse_srt(raw)
    if suffix in _TEXT_SUFFIXES:
        return ingest_text(_read_text(p))
    raise IngestError(
        f"Unsupported file type '{suffix or '(none)'}'. "
        "Supported: .srt, .txt, .md (or paste text directly)."
    )


def ingest_srt_text(raw: str) -> Transcript:
    """Parse raw SRT-formatted text (e.g. captions fetched via ``yt-dlp``) into a Transcript."""
    return _parse_srt(raw)


def ingest_text(text: str) -> Transcript:
    """Normalize pasted/plain text. Detects inline ``HH:MM:SS`` and ``MM:SS`` markers per line."""
    if not text or not text.strip():
        raise IngestError("Empty input: nothing to ingest.")

    # If most non-blank lines start with an inline timestamp, treat line-per-segment.
    # HH:MM:SS is checked first since its stricter shape can't collide with MM:SS's.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hhmmss_lines = [ln for ln in lines if _INLINE_TS_HHMMSS.match(ln)]
    if lines and len(hhmmss_lines) >= max(1, len(lines) // 2):
        transcript = _parse_inline_timestamped(text)
    else:
        mmss_lines = [ln for ln in lines if _INLINE_TS_MMSS.match(ln)]
        if lines and len(mmss_lines) >= max(1, len(lines) // 2):
            transcript = _parse_mmss_rolling_caption(text)
        else:
            transcript = _parse_paragraphs(text)
    _check_min_words(transcript)
    return transcript


# ---- format parsers ---------------------------------------------------------------------


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise IngestError(f"Could not read {p} as UTF-8 text (binary file?).") from exc


def _parse_srt(raw: str) -> Transcript:
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", raw.strip())
    idx = 0
    for block in blocks:
        block_lines = [ln for ln in block.splitlines() if ln.strip()]
        if not block_lines:
            continue
        # Optional leading numeric index line.
        if _SRT_INDEX_ONLY.match(block_lines[0].strip()):
            block_lines = block_lines[1:]
        if not block_lines:
            continue
        timestamp = None
        m = _SRT_TIME.search(block_lines[0])
        if m:
            timestamp = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
            block_lines = block_lines[1:]
        text = " ".join(block_lines).strip()
        if not text:
            continue
        segments.append(Segment(text=text, locator=f"seg:{idx}", timestamp=timestamp))
        idx += 1
    if not segments:
        raise IngestError("No subtitle cues found in .srt input.")
    transcript = Transcript(segments=segments)
    _check_min_words(transcript)
    return transcript


def _parse_inline_timestamped(text: str) -> Transcript:
    segments: list[Segment] = []
    idx = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _INLINE_TS_HHMMSS.match(line)
        if m:
            ts, body = m.group(1), m.group(2).strip()
            ts = _normalize_ts(ts)
            if body:
                segments.append(Segment(text=body, locator=f"seg:{idx}", timestamp=ts))
                idx += 1
        else:
            segments.append(Segment(text=line.strip(), locator=f"seg:{idx}", timestamp=None))
            idx += 1
    if not segments:
        raise IngestError("No usable lines found in input.")
    return Transcript(segments=segments)


def _parse_mmss_rolling_caption(text: str) -> Transcript:
    """Parse a YouTube-transcript-panel-style ``MM:SS`` caption dump.

    Called only after ``ingest_text`` has already seen a majority of non-blank lines open
    with the ``MM:SS`` shape. From here the bar for actually trusting that shape is higher and
    absolute, not majority: every non-blank line must match, and the matched timestamps must
    never decrease — either failing means this isn't confidently a caption export (could be
    prose that coincidentally opens several lines with something clock-like), so this refuses
    with IngestError instead of guessing which lines are real captions.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    parsed: list[tuple[int, str]] = []
    for ln in lines:
        m = _INLINE_TS_MMSS.match(ln)
        if not m:
            raise IngestError(
                "Input looks like MM:SS-timestamped captions, but line "
                f"{ln.strip()!r} doesn't match that shape — refusing rather than "
                "guessing which lines are captions."
            )
        minutes, seconds = m.group(1).split(":")
        parsed.append((int(minutes) * 60 + int(seconds), m.group(2).strip()))

    for (prev_ts, _), (cur_ts, _) in zip(parsed, parsed[1:], strict=False):
        if cur_ts < prev_ts:
            raise IngestError(
                "Input has MM:SS-shaped line starts, but the timestamps are not "
                "in non-decreasing order — this doesn't look like real caption "
                "output, refusing rather than guessing."
            )

    segments: list[Segment] = []
    idx = 0
    for total_seconds, body in parsed:
        # A rolling-caption window previews its next timestamp on a text-less line just
        # before the real text for it appears; that preview carries no speech and is noise.
        if not body:
            continue
        segments.append(
            Segment(text=body, locator=f"seg:{idx}", timestamp=_seconds_to_hhmmss(total_seconds))
        )
        idx += 1
    if not segments:
        raise IngestError("No usable caption lines found in input.")
    return Transcript(segments=segments)


def _seconds_to_hhmmss(total_seconds: int) -> str:
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_paragraphs(text: str) -> Transcript:
    """Split on blank lines into paragraphs; drop pure markdown headings."""
    segments: list[Segment] = []
    idx = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        cleaned_lines = []
        for ln in block.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue  # markdown heading: not a knowledge segment
            cleaned_lines.append(stripped)
        body = " ".join(cleaned_lines).strip()
        if body:
            segments.append(Segment(text=body, locator=f"seg:{idx}", timestamp=None))
            idx += 1
    if not segments:
        raise IngestError("Input contained no extractable text (only headings/blank lines?).")
    return Transcript(segments=segments)


def _normalize_ts(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return ts
