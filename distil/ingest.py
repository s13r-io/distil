"""Stage 0 — Ingest (PURE, no LLM). ARCHITECTURE.md §2; PRD FR21/FR22; TESTING T-I*.

Parses any supported input — ``.srt``, ``.txt``, ``.md``, or pasted text — into one
normalized :class:`Transcript`: an ordered list of :class:`Segment` ``{text, timestamp?,
locator}``. Timestamps are captured when the source has them (SRT cues or inline
``HH:MM:SS`` markers) and left ``None`` otherwise; a ``locator`` (``seg:<index>``) is always
populated so untimestamped sources still have a stable pointer. Downstream stages depend only
on this shape, never on the original format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_INLINE_TS = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2})(?:[.,]\d+)?\s+(.*)$")
_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,]\d{3}\s*-->")
_SRT_INDEX_ONLY = re.compile(r"^\d+$")

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".vtt"}


class IngestError(ValueError):
    """Raised for empty input, a missing file, or an unsupported/binary format."""


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


def ingest_text(text: str) -> Transcript:
    """Normalize pasted/plain text. Detects inline ``HH:MM:SS`` markers per line."""
    if not text or not text.strip():
        raise IngestError("Empty input: nothing to ingest.")

    # If most non-blank lines start with an inline timestamp, treat line-per-segment.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    ts_lines = [ln for ln in lines if _INLINE_TS.match(ln)]
    if lines and len(ts_lines) >= max(1, len(lines) // 2):
        return _parse_inline_timestamped(text)

    return _parse_paragraphs(text)


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
    return Transcript(segments=segments)


def _parse_inline_timestamped(text: str) -> Transcript:
    segments: list[Segment] = []
    idx = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _INLINE_TS.match(line)
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
