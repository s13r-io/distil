"""Phase D — entities are retrievable under the exact same relevance threshold as items and
concepts (query.py's retrieve_entities / ask / stream_ask). Mirrors test_query.py's T-Q9/T-Q11/
T-Q13 concept tests one granularity down.
"""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Entity, EntityClaim, EntityMember, KBEntry
from distil.query import ask, retrieve_entities
from distil.store import Store


def _entry(entry_id, items) -> KBEntry:
    data = {
        "entry_id": entry_id,
        "source": {"title": entry_id, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [
            {"item_id": iid, "type": "heuristic", "statement": stmt, "stance": "opinion",
             "provenance": {"quote": stmt[:20].lower(), "timestamp": ts}}
            for iid, stmt, ts in items
        ],
        "tags": {"topics": [], "knowledge_types": ["heuristic"], "application_forms": []},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    }
    return KBEntry.model_validate(data)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    s.file_entry(_entry("e_react", [
        ("k_r1", "React renders UI using a virtual DOM.", "00:01:00"),
    ]), embedder=emb)
    s.file_entry(_entry("e_k8s", [
        ("k_k1", "Kubernetes pods share a network namespace.", "00:05:00"),
    ]), embedder=emb)
    return s


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=64)


def _entity(entity_id, kind, title, description, members, claims=None) -> Entity:
    return Entity(
        entity_id=entity_id, kind=kind, title=title, description=description,
        members=members, claims=claims or [],
        created_at="2026-06-15T00:00:00", updated_at="2026-06-15T00:00:00",
    )


@pytest.mark.unit
def test_retrieve_entities_ranks_relevant_first(store, embedder):
    react_entity = _entity(
        "react", "tool", "React", "A JS UI library rendering via a virtual DOM.",
        members=[EntityMember(entry_id="e_react", item_id="k_r1", quote="q", timestamp=None)],
    )
    k8s_entity = _entity(
        "kubernetes", "tool", "Kubernetes", "Container orchestration with shared pod networking.",
        members=[EntityMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
    )
    store.save_entity(react_entity)
    store.save_entity(k8s_entity)

    results = retrieve_entities("virtual DOM rendering", store, embedder, top_k=2)
    assert results[0].entity_id == "react"
    assert results[0].similarity >= results[-1].similarity


@pytest.mark.unit
def test_entity_only_evidence_avoids_abstention_under_same_threshold(store, embedder, monkeypatch):
    """An entity whose centroid clears `threshold` on its own recruits its members into the
    evidence pool and avoids abstention — the same gate concepts and raw items share (T-Q11,
    one granularity down). No easier bar for entities."""
    entity = _entity(
        "kubernetes", "tool", "Kubernetes", "desc",
        members=[EntityMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
        claims=[EntityClaim(text="Kubernetes pods share a network namespace.", item_ids=["k_k1"])],
    )
    store.save_entity(entity)

    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    qvec = embedder.embed(question)
    monkeypatch.setattr(store, "entity_centroid", lambda entity_id: qvec)

    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])
    result = ask(question, store, embedder, client, threshold=0.99, top_k=3)

    assert not result.abstained
    assert any(s.item_id == "k_k1" for s in result.sources)
    assert "k_k1" in result.cited_item_ids


@pytest.mark.unit
def test_low_similarity_entity_does_not_bypass_gate(store, embedder):
    entity = _entity(
        "kubernetes", "tool", "Kubernetes", "desc",
        members=[EntityMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
    )
    store.save_entity(entity)

    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask(
        "medieval french history and gothic cathedrals", store, embedder, client, threshold=0.9,
    )
    assert result.abstained
    assert client.call_count == 0


@pytest.mark.unit
def test_entity_answer_sources_are_code_assembled_from_real_provenance(store, embedder, monkeypatch):
    """The EntityMember's copied quote/timestamp deliberately mismatch the real item's — a
    passing assertion proves Source data was resolved fresh from the live KBEntry, never trusted
    from the entity's own copy (mirrors T-Q12 for concepts)."""
    entity = _entity(
        "kubernetes", "tool", "Kubernetes", "desc",
        members=[EntityMember(entry_id="e_k8s", item_id="k_k1", quote="stale-copy", timestamp="99:99:99")],
    )
    store.save_entity(entity)

    question = "another unrelated phrase with no vocabulary overlap"
    qvec = embedder.embed(question)
    monkeypatch.setattr(store, "entity_centroid", lambda entity_id: qvec)

    result = ask(
        question, store, embedder, FakeClient(responses=["SHOULD NOT BE CALLED"]),
        threshold=0.99, top_k=3, lookup_only=True,
    )
    assert not result.abstained
    real_item = next(i for i in store.load_entry("e_k8s").knowledge_items if i.item_id == "k_k1")
    src = next(s for s in result.sources if s.item_id == "k_k1")
    assert src.quote == real_item.provenance.quote
    assert src.quote != "stale-copy"
