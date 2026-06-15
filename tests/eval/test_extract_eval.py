"""Phase 4.3 — FAITHFULNESS eval. Test T-E3 (the headline guarantee).

Every returned item's provenance quote must appear verbatim in the source transcript, for
timestamped AND untimestamped sources alike. Zero tolerance for fabricated provenance.

Marked ``eval``: requires ANTHROPIC_API_KEY + DISTIL_MODEL, skipped in normal CI.
"""

from pathlib import Path

import pytest

from distil.extract import run_extraction
from distil.faithfulness import quote_in_transcript
from distil.ingest import ingest_file
from distil.llm import AnthropicClient
from distil.triage import run_triage

FIX = Path(__file__).parent.parent / "fixtures"

# Mix of timestamped and untimestamped sources — faithfulness must hold for both.
_SOURCES = [
    "rich_heuristic.txt",
    "procedural_tutorial.txt",
    "mixed_talk.txt",
    "sample.srt",
    "no_timestamps.md",
]


@pytest.mark.eval
@pytest.mark.parametrize("fixture", _SOURCES)
def test_e3_every_quote_appears_in_transcript(fixture):
    transcript = ingest_file(FIX / fixture)
    client = AnthropicClient()
    triage = run_triage(transcript, client)
    if triage.triage.verdict == "little_to_extract":
        pytest.skip(f"{fixture} triaged as low-value; nothing to extract")
    items = run_extraction(transcript, triage.triage, client)
    for item in items:
        assert quote_in_transcript(item.provenance.quote, transcript), (
            f"FABRICATED PROVENANCE in {fixture}: quote not found in transcript: "
            f"{item.provenance.quote!r}"
        )
