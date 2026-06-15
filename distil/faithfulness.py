"""The faithfulness gate (deterministic, PURE).

A knowledge item is faithful iff its ``provenance.quote`` appears verbatim in the source
transcript — the format-independent anchor behind T-E3 (eval) and T-N2 (normalize drops
unverifiable items). Matching tolerates differences that don't change the words: case,
runs of whitespace (including the newlines ``full_text`` inserts between segments), and
surrounding punctuation. It does NOT tolerate different words — that is fabrication.
"""

from __future__ import annotations

import re

from .ingest import Transcript

# Keep word characters and spaces; turn everything else (punctuation) into a space.
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    lowered = text.lower()
    no_punct = _NON_WORD.sub(" ", lowered)
    return _WS.sub(" ", no_punct).strip()


def quote_in_transcript(quote: str, transcript: Transcript) -> bool:
    """True iff ``quote`` (normalized) is a contiguous substring of the transcript text."""
    needle = _normalize(quote)
    if not needle:
        return False
    haystack = _normalize(transcript.full_text())
    return needle in haystack
