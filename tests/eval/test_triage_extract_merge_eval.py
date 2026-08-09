"""Owner decision (addendum): triage and extraction are merged into one strong-tier call
(``extract.run_triage_extract``) instead of two separate calls. Additional acceptance for that
change:

- The strong model reads the full transcript exactly once per ingest — proved by call counting
  in ``tests/unit/test_pipeline_summary.py`` (unit, FakeClient), not repeated here.
- The merged call still yields the dominant type, and extraction is still conditioned on it.
- Extraction quality does not regress relative to the old split (triage call, then extract
  call) path.

This file covers the two quality-facing properties that need a real model: faithfulness on the
merged path (mirrors T-E3 for `run_extraction`), and a same-fixture comparison against the split
path, reporting counts/types honestly rather than asserting exact equality (the two calls are
not expected to produce byte-identical output — same model, different prompt shape — only
comparable coverage).

Marked ``eval``: requires ANTHROPIC_API_KEY + DISTIL_MODEL, skipped in normal CI. NOTE: this
suite could not be executed in the environment this change was authored in (no API key
available there) — see the PR description for that limitation stated plainly rather than a
fabricated result. Run `pytest -m eval -k triage_extract_merge` with real credentials before
relying on this acceptance criterion being met.
"""

from pathlib import Path

import pytest

from distil.extract import run_extraction, run_triage_extract
from distil.faithfulness import quote_in_transcript
from distil.ingest import ingest_file
from distil.llm import AnthropicClient
from distil.triage import run_triage

FIX = Path(__file__).parent.parent / "fixtures"

_SOURCES = [
    "rich_heuristic.txt",
    "procedural_tutorial.txt",
    "mixed_talk.txt",
    "sample.srt",
    "no_timestamps.md",
]


@pytest.mark.eval
@pytest.mark.parametrize("fixture", _SOURCES)
def test_merged_call_every_quote_appears_in_transcript(fixture):
    """Faithfulness must hold identically on the merged path — the headline guarantee (T-E3)
    does not get to regress just because triage and extraction now share one call."""
    transcript = ingest_file(FIX / fixture)
    client = AnthropicClient()
    result = run_triage_extract(transcript, client)
    for item in result.items:
        assert quote_in_transcript(item.provenance.quote, transcript), (
            f"FABRICATED PROVENANCE in {fixture} (merged call): quote not found in "
            f"transcript: {item.provenance.quote!r}"
        )


@pytest.mark.eval
@pytest.mark.parametrize("fixture", _SOURCES)
def test_merged_call_extraction_is_conditioned_on_its_own_stated_dominant_type(fixture):
    """The "decide-then-act" property the merge is required to preserve: the dominant type the
    merged call stated in its own TRIAGE section must actually show up among the item types it
    extracted (when it extracted anything at all) — proof the ITEMS section was genuinely
    conditioned on the classification that preceded it in the same response, not independent of
    it."""
    transcript = ingest_file(FIX / fixture)
    client = AnthropicClient()
    result = run_triage_extract(transcript, client)
    if not result.items:
        pytest.skip(f"{fixture}: merged call extracted nothing to check conditioning against")
    from distil.extract import dominant_type

    dominant = dominant_type(result.triage)
    item_types = {item.type for item in result.items}
    assert dominant in item_types, (
        f"{fixture}: merged call's stated dominant type {dominant!r} is absent from its own "
        f"extracted item types {item_types!r} — looks decided independently, not conditioned"
    )


@pytest.mark.eval
@pytest.mark.parametrize("fixture", _SOURCES)
def test_merged_call_coverage_is_comparable_to_the_split_path(fixture):
    """Not a strict equality check (same model, different prompt shape — exact item-for-item
    parity across two separate completions was never realistic), but a coverage floor: the
    merged call must not systematically extract dramatically less than the split path did, on
    the same transcript. Report the actual counts either way so a real regression is visible in
    the eval output, not just a pass/fail bit.
    """
    transcript = ingest_file(FIX / fixture)
    client = AnthropicClient()

    split_triage = run_triage(transcript, client)
    if split_triage.triage.verdict == "little_to_extract":
        pytest.skip(f"{fixture}: triaged as low-value on the split path; nothing to compare")
    split_items = run_extraction(transcript, split_triage.triage, client)

    merged = run_triage_extract(transcript, client)

    print(
        f"\n[{fixture}] split={len(split_items)} items "
        f"({sorted({i.type for i in split_items})}), "
        f"merged={len(merged.items)} items ({sorted({i.type for i in merged.items})})"
    )

    # A generous floor — this is a coverage sanity check, not a precision benchmark. A merged
    # call producing zero items against a non-trivial split result is the one shape worth
    # failing loudly on; smaller variance either way is expected model noise.
    if split_items:
        assert len(merged.items) > 0, (
            f"{fixture}: split path found {len(split_items)} items; merged call found none — "
            "this is the regression this test exists to catch"
        )
