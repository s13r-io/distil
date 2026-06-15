"""Phase 3.4 — triage eval against fixtures. Tests T-T4, T-T5.

Marked ``eval``: requires ANTHROPIC_API_KEY + DISTIL_MODEL, skipped in normal CI. Asserts
*properties* (the verdict, the loss level), not exact strings.
"""

import json
from pathlib import Path

import pytest

from distil.ingest import ingest_file
from distil.llm import AnthropicClient
from distil.triage import run_triage

FIX = Path(__file__).parent.parent / "fixtures"


def _expected(name: str) -> dict:
    return json.loads((FIX / f"{name}.expected.json").read_text())


@pytest.mark.eval
def test_t4_low_value_vlog_yields_little_to_extract():
    transcript = ingest_file(FIX / "low_value_vlog.txt")
    result = run_triage(transcript, AnthropicClient())
    assert result.triage.verdict == _expected("low_value_vlog")["verdict"]


@pytest.mark.eval
def test_t5_screen_share_high_loss_with_evidence():
    transcript = ingest_file(FIX / "screen_share.txt")
    result = run_triage(transcript, AnthropicClient())
    assert result.triage.transcript_loss.level == _expected("screen_share")["transcript_loss"]
    assert len(result.triage.transcript_loss.evidence) > 0


@pytest.mark.eval
def test_rich_heuristic_is_not_low_value():
    transcript = ingest_file(FIX / "rich_heuristic.txt")
    result = run_triage(transcript, AnthropicClient())
    assert result.triage.verdict != "little_to_extract"
