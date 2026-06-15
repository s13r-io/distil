"""Phase 10.2/10.3 — vector storage in store.py. Tests T-X1, T-X2.

Uses FakeEmbedder so tests are hermetic. The vector backend is abstracted: sqlite-vec when
available, else a pure-Python fallback table — same API either way.
"""

import pytest

from distil.embed import FakeEmbedder
from distil.models import KBEntry
from distil.store import Store


def _entry(entry_id="e_01", statements=("Keep functions small.",)) -> KBEntry:
    items = [
        {
            "item_id": f"k_{i:02d}", "type": "heuristic", "statement": s,
            "stance": "opinion", "provenance": {"quote": s[:20].lower()},
        }
        for i, s in enumerate(statements, start=1)
    ]
    return KBEntry.model_validate({
        "entry_id": entry_id,
        "source": {"title": entry_id, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": items,
        "tags": {"topics": [], "knowledge_types": ["heuristic"], "application_forms": []},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    })


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


# ---- T-X1: filing an entry stores one vector per knowledge item ----


@pytest.mark.unit
def test_x1_file_with_embedder_stores_one_vector_per_item(store):
    embedder = FakeEmbedder(dim=16)
    store.file_entry(_entry(statements=("A.", "B.", "C.")), embedder=embedder)
    assert store.vector_count() == 3


@pytest.mark.unit
def test_x1_vectors_carry_item_and_entry_fk(store):
    embedder = FakeEmbedder(dim=16)
    store.file_entry(_entry(entry_id="e_42", statements=("only one.",)), embedder=embedder)
    rows = store.all_vector_rows()
    assert len(rows) == 1
    assert rows[0].entry_id == "e_42"
    assert rows[0].item_id == "k_01"
    assert rows[0].embedding_model == embedder.model_name


@pytest.mark.unit
def test_file_without_embedder_stores_no_vectors(store):
    store.file_entry(_entry())
    assert store.vector_count() == 0


# ---- T-X2: reindex backfills; idempotent ----


@pytest.mark.unit
def test_x2_reindex_backfills_entries_filed_without_vectors(store):
    store.file_entry(_entry(entry_id="e_01", statements=("a.", "b.")))  # no embedder
    assert store.vector_count() == 0
    n = store.reindex(FakeEmbedder(dim=16))
    assert n == 2
    assert store.vector_count() == 2


@pytest.mark.unit
def test_x2_reindex_is_idempotent(store):
    embedder = FakeEmbedder(dim=16)
    store.file_entry(_entry(entry_id="e_01", statements=("a.", "b.")), embedder=embedder)
    assert store.vector_count() == 2
    added = store.reindex(embedder)  # already embedded → nothing new
    assert added == 0
    assert store.vector_count() == 2


@pytest.mark.unit
def test_x2_reindex_after_model_change_reembeds(store):
    store.file_entry(_entry(entry_id="e_01", statements=("a.",)), embedder=FakeEmbedder(dim=8))
    assert store.vector_count() == 1
    # different model name → reindex replaces stale vectors, not duplicate
    added = store.reindex(FakeEmbedder(dim=16))
    assert added == 1
    assert store.vector_count() == 1


@pytest.mark.unit
def test_vectors_persist_across_instances(tmp_path):
    db, kb = tmp_path / "d.db", tmp_path / "kb"
    Store(db_path=db, kb_dir=kb).file_entry(_entry(statements=("x.",)), embedder=FakeEmbedder(dim=8))
    assert Store(db_path=db, kb_dir=kb).vector_count() == 1
