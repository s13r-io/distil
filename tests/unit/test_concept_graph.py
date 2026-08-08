"""Phase 16 — concept_graph.py: concept<->concept typed edges. Design report §9 item 4.

Mirrors test_graph.py's T-G1/T-G2 shape, applied at concept granularity: candidate lookup is
deterministic (centroid cosine similarity), the LLM only labels the relationship.
"""

import json

import pytest

from distil.concept_graph import link_concept_graph, run_concept_edges_stage
from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Concept, ConceptMember, KBEntry
from distil.store import Store


def _entry(entry_id, item_id, statement, quote="q") -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": f"Video {entry_id}", "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": [
                {
                    "item_id": item_id,
                    "type": "conceptual",
                    "statement": statement,
                    "stance": "fact",
                    "provenance": {"quote": quote},
                }
            ],
            "tags": {"topics": ["rag"], "knowledge_types": ["conceptual"], "application_forms": []},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _concept(concept_id, title, entry_id, item_id, quote="q") -> Concept:
    return Concept(
        concept_id=concept_id,
        title=title,
        description=f"{title} description.",
        members=[ConceptMember(entry_id=entry_id, item_id=item_id, quote=quote)],
        claims=[],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


def _file_and_save(store, embedder, entry, concept):
    store.file_entry(entry, embedder=embedder)
    store.save_concept(concept)  # recomputes + persists the centroid from stored item vectors


# ---- candidate lookup is deterministic (no LLM when no other concepts exist) ----


@pytest.mark.unit
def test_no_candidates_makes_no_llm_call(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    concept = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    _file_and_save(store, embedder, e1, concept)

    fake = FakeClient(responses=['{"relation": "related"}'])
    edges = link_concept_graph(concept, store, fake)

    assert edges == []
    assert fake.call_count == 0


@pytest.mark.unit
def test_similar_concept_becomes_a_candidate_and_is_classified(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    e2 = _entry("e_02", "k_02", "Traditional RAG retrieves then generates too")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c2 = _concept("agentic-rag", "Agentic RAG", "e_02", "k_02")
    _file_and_save(store, embedder, e1, c1)
    _file_and_save(store, embedder, e2, c2)

    fake = FakeClient(responses=['{"relation": "builds_on"}'])
    edges = link_concept_graph(c1, store, fake)

    assert len(edges) == 1
    assert edges[0].target_concept_id == "agentic-rag"
    assert edges[0].relation == "builds_on"


# ---- relation classification maps to the allowed enum only ----


@pytest.mark.unit
def test_none_relation_is_dropped(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    e2 = _entry("e_02", "k_02", "Traditional RAG retrieves then generates too")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c2 = _concept("agentic-rag", "Agentic RAG", "e_02", "k_02")
    _file_and_save(store, embedder, e1, c1)
    _file_and_save(store, embedder, e2, c2)

    fake = FakeClient(responses=[json.dumps({"relation": "none"})])
    edges = link_concept_graph(c1, store, fake)

    assert edges == []


@pytest.mark.unit
def test_invalid_relation_is_dropped_not_crash(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    e2 = _entry("e_02", "k_02", "Traditional RAG retrieves then generates too")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c2 = _concept("agentic-rag", "Agentic RAG", "e_02", "k_02")
    _file_and_save(store, embedder, e1, c1)
    _file_and_save(store, embedder, e2, c2)

    fake = FakeClient(responses=['{"relation": "frenemy"}'])
    edges = link_concept_graph(c1, store, fake)

    assert edges == []


# ---- candidate pool is capped at MAX_CONCEPT_EDGE_CANDIDATES ----


@pytest.mark.unit
def test_candidate_pool_is_capped(store, embedder, monkeypatch):
    monkeypatch.delenv("DISTIL_CONCEPT_SIM_FLOOR", raising=False)
    e0 = _entry("e_00", "k_00", "Traditional RAG retrieves then generates")
    c0 = _concept("traditional-rag", "Traditional RAG", "e_00", "k_00")
    _file_and_save(store, embedder, e0, c0)

    for i in range(5):
        entry = _entry(f"e_c{i}", "k_01", "Traditional RAG retrieves then generates")
        concept = _concept(f"concept-{i}", f"Concept {i}", f"e_c{i}", "k_01")
        _file_and_save(store, embedder, entry, concept)

    fake = FakeClient(
        responses=[json.dumps({"relation": "related"})] * 3  # MAX_CONCEPT_EDGE_CANDIDATES
    )
    edges = link_concept_graph(c0, store, fake)

    assert fake.call_count == 3
    assert len(edges) == 3


# ---- run_concept_edges_stage: pending concepts are skipped; export happens for changed ones ----


@pytest.mark.unit
def test_stage_skips_pending_synthesis_concepts(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c1.pending_synthesis = True
    _file_and_save(store, embedder, e1, c1)

    fake = FakeClient(responses=[])
    changed = run_concept_edges_stage([c1], store, fake)

    assert changed == []
    assert fake.call_count == 0
    saved = store.load_concept("traditional-rag")
    assert saved.edges == []


@pytest.mark.unit
def test_stage_computes_and_exports_edges_for_ready_concepts(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    e2 = _entry("e_02", "k_02", "Traditional RAG retrieves then generates too")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c2 = _concept("agentic-rag", "Agentic RAG", "e_02", "k_02")
    _file_and_save(store, embedder, e1, c1)
    _file_and_save(store, embedder, e2, c2)

    fake = FakeClient(responses=[json.dumps({"relation": "related"})])
    changed = run_concept_edges_stage([c1], store, fake)

    assert {c.concept_id for c in changed} == {"traditional-rag"}
    saved = store.load_concept("traditional-rag")
    assert saved.edges[0].target_concept_id == "agentic-rag"
    page = store.okf_root / "concepts" / "traditional-rag.md"
    assert page.exists()
    assert "## Related" in page.read_text()
    assert "[Agentic RAG](agentic-rag.md)" in page.read_text()


@pytest.mark.unit
def test_stage_prunes_dangling_edges_to_deleted_concepts(store, embedder):
    e1 = _entry("e_01", "k_01", "Traditional RAG retrieves then generates")
    e2 = _entry("e_02", "k_02", "Traditional RAG retrieves then generates too")
    c1 = _concept("traditional-rag", "Traditional RAG", "e_01", "k_01")
    c2 = _concept("agentic-rag", "Agentic RAG", "e_02", "k_02")
    _file_and_save(store, embedder, e1, c1)
    _file_and_save(store, embedder, e2, c2)
    run_concept_edges_stage([c1], store, FakeClient(responses=[json.dumps({"relation": "related"})]))
    assert store.load_concept("traditional-rag").edges != []

    store.delete_concept("agentic-rag")
    changed = run_concept_edges_stage([], store, FakeClient(responses=[]))

    assert {c.concept_id for c in changed} == {"traditional-rag"}
    assert store.load_concept("traditional-rag").edges == []
    page = (store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert "## Related" not in page
