"""Phase 5.1 — normalize.py (PURE). Tests T-N1..N4.

A deterministic gate after extraction: merge near-duplicates, DROP items whose provenance
quote is not in the transcript (never weaken this), preserve stance, and attach a locator for
untimestamped sources.
"""

import pytest

from distil.ingest import ingest_text
from distil.models import KnowledgeItem
from distil.normalize import normalize_items


def _item(item_id, statement, quote, *, stance="fact", ktype="declarative",
          timestamp=None, locator=None) -> KnowledgeItem:
    return KnowledgeItem.model_validate({
        "item_id": item_id,
        "type": ktype,
        "statement": statement,
        "stance": stance,
        "provenance": {"quote": quote, "timestamp": timestamp, "locator": locator},
    })


# ---- T-N1: near-duplicate items are merged ----


@pytest.mark.unit
def test_near_duplicates_merged():
    t = ingest_text("Keep functions small. Name things clearly.")
    items = [
        _item("k_01", "Keep functions small.", "keep functions small"),
        _item("k_02", "Keep functions  small!", "keep functions small"),  # near-dup
        _item("k_03", "Name things clearly.", "name things clearly"),
    ]
    out = normalize_items(items, t)
    statements = {i.statement for i in out}
    assert len(out) == 2
    assert any("functions" in s for s in statements)
    assert any("Name things" in s for s in statements)


# ---- T-N2: item whose provenance quote is NOT in the transcript is dropped ----


@pytest.mark.unit
def test_unverifiable_provenance_dropped():
    t = ingest_text("Keep functions small.")
    items = [
        _item("k_01", "Keep functions small.", "keep functions small"),
        _item("k_02", "Always use microservices.", "always use microservices"),  # fabricated
    ]
    out = normalize_items(items, t)
    assert len(out) == 1
    assert out[0].item_id == "k_01"


@pytest.mark.unit
def test_all_unverifiable_yields_empty():
    t = ingest_text("Keep functions small.")
    items = [_item("k_01", "Invented.", "this was never said")]
    assert normalize_items(items, t) == []


# ---- T-N3: opinion content keeps stance == opinion; never rewritten to look like fact ----


@pytest.mark.unit
def test_stance_preserved():
    t = ingest_text("I think microservices are usually overkill for small teams.")
    items = [
        _item("k_01", "Microservices are usually overkill for small teams.",
              "microservices are usually overkill", stance="opinion", ktype="opinion"),
    ]
    out = normalize_items(items, t)
    assert out[0].stance == "opinion"


# ---- T-N4: untimestamped source → items validate with timestamp=null + populated locator ----


@pytest.mark.unit
def test_untimestamped_gets_locator_and_passes_quote_gate():
    t = ingest_text("Deliberate practice means working at the edge of your ability.")
    items = [
        _item("k_01", "Practice at the edge of your ability.",
              "working at the edge of your ability", timestamp=None, locator=None),
    ]
    out = normalize_items(items, t)
    assert len(out) == 1
    assert out[0].provenance.timestamp is None
    assert out[0].provenance.locator is not None  # backfilled from the matching segment


@pytest.mark.unit
def test_timestamped_locator_preserved():
    t = ingest_text("00:12:30 Never weaken a guarantee just to make a test pass.")
    items = [
        _item("k_01", "Don't weaken guarantees to pass tests.",
              "never weaken a guarantee", timestamp=None, locator=None),
    ]
    out = normalize_items(items, t)
    assert len(out) == 1
    # the matching segment carried a timestamp; normalize backfills it
    assert out[0].provenance.timestamp == "00:12:30"


@pytest.mark.unit
def test_pure_does_not_mutate_input():
    t = ingest_text("Keep functions small.")
    items = [_item("k_01", "Keep functions small.", "keep functions small")]
    before = items[0].model_dump_json()
    normalize_items(items, t)
    assert items[0].model_dump_json() == before
