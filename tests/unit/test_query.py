"""Phase 10.4-10.6 — retrieval, abstention gate, grounded synthesis.

Tests T-Q1..Q6. The headline guarantees:
* T-Q2 ABSTENTION: below-threshold question returns "no relevant notes" AND makes zero
  synthesis LLM calls.
* T-Q3 GROUNDING: an answered question cites only items from the retrieved set.
"""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.query import ask, retrieve
from distil.store import Store


def _entry(entry_id, items, *, score=None, related=None) -> KBEntry:
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
    if score is not None:
        data["feedback"] = {"score": score}
    if related:
        data["related_entries"] = related
    return KBEntry.model_validate(data)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    s.file_entry(_entry("e_py", [
        ("k_py1", "Write unit tests before the implementation code.", "00:01:00"),
        ("k_py2", "Keep python functions small and focused.", None),
    ]), embedder=emb)
    s.file_entry(_entry("e_k8s", [
        ("k_k1", "Kubernetes pods share a network namespace.", "00:05:00"),
    ]), embedder=emb)
    return s


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=64)


# ---- T-Q1: KNN ranked by similarity × feedback_score × recency ----


@pytest.mark.unit
def test_q1_retrieval_ranks_relevant_first(store, embedder):
    results = retrieve("python unit tests", store, embedder, top_k=3)
    assert results
    # the python testing note should outrank the kubernetes note
    ids = [r.item_id for r in results]
    assert ids[0] in {"k_py1", "k_py2"}
    assert results[0].score >= results[-1].score  # sorted descending


@pytest.mark.unit
def test_q1_feedback_score_boosts_ranking(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    # two entries with identical text; one is highly rated
    s.file_entry(_entry("e_low", [("k_low", "alpha beta gamma delta", None)], score=1), embedder=emb)
    s.file_entry(_entry("e_high", [("k_high", "alpha beta gamma delta", None)], score=5), embedder=emb)
    results = retrieve("alpha beta gamma delta", s, emb, top_k=2)
    assert results[0].item_id == "k_high"


# ---- T-Q2: ABSTENTION (headline) ----


@pytest.mark.unit
def test_q2_below_threshold_abstains_no_llm_call(store, embedder):
    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask("medieval french history and gothic cathedrals", store, embedder, client,
                 threshold=0.9)  # high threshold → nothing clears it
    assert result.abstained is True
    assert result.answer is None
    assert "no relevant notes" in result.message.lower()
    # THE GUARANTEE: synthesis LLM was never invoked
    assert client.call_count == 0


@pytest.mark.unit
def test_q2_threshold_zero_does_not_force_answer_on_empty_kb(tmp_path, embedder):
    empty = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    client = FakeClient(responses=["nope"])
    result = ask("anything", empty, embedder, client, threshold=0.0)
    assert result.abstained is True
    assert client.call_count == 0


# ---- T-Q3: GROUNDING (headline) ----


@pytest.mark.unit
def test_q3_answer_cites_only_retrieved_items(store, embedder):
    synth = json.dumps({
        "answer": "Write tests first [k_py1] and keep functions small [k_py2].",
        "cited_item_ids": ["k_py1", "k_py2"],
        "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("how should I write python code", store, embedder, client, threshold=0.0, top_k=3)
    assert not result.abstained
    retrieved_ids = {s.item_id for s in result.sources}
    assert set(result.cited_item_ids) <= retrieved_ids  # no citation outside retrieved set


@pytest.mark.unit
def test_q3_citations_outside_retrieved_set_are_flagged(store, embedder):
    # model fabricates a citation to an item that wasn't retrieved
    synth = json.dumps({
        "answer": "Use microservices always [k_fake].",
        "cited_item_ids": ["k_fake"],
        "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("architecture advice", store, embedder, client, threshold=0.0, top_k=3)
    # the ungrounded citation must not be presented as a valid source
    retrieved_ids = {s.item_id for s in result.sources}
    assert "k_fake" not in retrieved_ids
    assert result.ungrounded_citations == ["k_fake"]


# ---- T-Q4: every answer carries resolvable source links ----


@pytest.mark.unit
def test_q4_sources_resolve_to_entry_item_provenance(store, embedder):
    synth = json.dumps({
        "answer": "Write tests first [k_py1].", "cited_item_ids": ["k_py1"], "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("testing", store, embedder, client, threshold=0.0, top_k=3)
    src = next(s for s in result.sources if s.item_id == "k_py1")
    assert src.entry_id == "e_py"
    assert src.timestamp == "00:01:00"
    assert src.quote


# ---- T-Q5: bare lookup returns ranked sources, no synthesis call ----


@pytest.mark.unit
def test_q5_lookup_returns_sources_without_synthesis(store, embedder):
    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask("python tests", store, embedder, client, threshold=0.0, top_k=3, lookup_only=True)
    assert not result.abstained
    assert result.answer is None
    assert len(result.sources) > 0
    assert client.call_count == 0


# ---- T-Q6: conflict surfaced when retrieved items contradict ----


@pytest.mark.unit
def test_q6_conflict_surfaced(tmp_path, embedder):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    s.file_entry(_entry("e_a", [("k_a", "monoliths are better for small teams indeed", None)]),
                 embedder=emb)
    s.file_entry(_entry("e_b", [("k_b", "monoliths are worse for small teams indeed", None)],
                        related=[{"target": "e_a", "relation": "contradicts"}]), embedder=emb)
    synth = json.dumps({
        "answer": "Notes disagree on monoliths [k_a][k_b].",
        "cited_item_ids": ["k_a", "k_b"],
        "conflict": "k_a says better, k_b says worse.",
    })
    client = FakeClient(responses=[synth])
    result = ask("are monoliths good for small teams", s, emb, client, threshold=0.0, top_k=3)
    assert result.conflict
