"""Faithfulness gate (deterministic). Underpins T-E3 (eval) and T-N2 (normalize drop)."""

import pytest

from distil.faithfulness import quote_in_transcript
from distil.ingest import Segment, Transcript


def _t(*texts: str) -> Transcript:
    return Transcript(
        segments=[Segment(text=text, locator=f"seg:{i}") for i, text in enumerate(texts)]
    )


@pytest.mark.unit
def test_exact_quote_matches():
    t = _t("Keep functions small and focused.")
    assert quote_in_transcript("keep functions small", t)


@pytest.mark.unit
def test_match_is_case_and_whitespace_insensitive():
    t = _t("Keep   functions    small.", "Name things clearly.")
    assert quote_in_transcript("KEEP functions small", t)
    assert quote_in_transcript("name things   clearly", t)


@pytest.mark.unit
def test_fabricated_quote_does_not_match():
    t = _t("Keep functions small and focused.")
    assert not quote_in_transcript("always use microservices", t)


@pytest.mark.unit
def test_match_spans_segment_join():
    # full_text joins segments with newline; matching ignores that boundary whitespace
    t = _t("first paragraph ends here", "second paragraph starts")
    assert quote_in_transcript("here second paragraph", t)


@pytest.mark.unit
def test_empty_quote_never_matches():
    t = _t("anything")
    assert not quote_in_transcript("", t)
    assert not quote_in_transcript("   ", t)


@pytest.mark.unit
def test_match_ignores_surrounding_punctuation():
    t = _t('He said, "write the test first," and meant it.')
    assert quote_in_transcript("write the test first", t)
