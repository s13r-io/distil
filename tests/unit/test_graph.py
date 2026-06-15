"""Phase 7 — graph.py. Tests T-G1 (deterministic candidate lookup), T-G2 (relation enum)."""

import json

import pytest

from distil.graph import link_graph
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.store import Store


def _entry(entry_id, topics, types=("heuristic",), statement="insight") -> KBEntry:
    return KBEntry.model_validate({
        "entry_id": entry_id,
        "source": {"title": entry_id, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01", "type": "heuristic", "statement": statement,
            "stance": "opinion", "provenance": {"quote": "q"},
        }],
        "tags": {"topics": list(topics), "knowledge_types": list(types), "application_forms": []},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    })


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    s.file_entry(_entry("e_old1", ["python", "testing"]))
    s.file_entry(_entry("e_old2", ["kubernetes"]))
    return s


# ---- T-G1: candidate lookup returns existing entries sharing topics/items (no LLM) ----


@pytest.mark.unit
def test_g1_candidate_lookup_is_deterministic_no_llm(store):
    new = _entry("e_new", ["python", "ci"])
    fake = FakeClient(responses=['{"relation": "supports"}'])
    edges = link_graph(new, store, fake)
    # only e_old1 shares a topic ("python"); e_old2 (kubernetes) is not a candidate
    assert {e.target for e in edges} == {"e_old1"}


@pytest.mark.unit
def test_g1_no_candidates_makes_no_llm_call(store):
    new = _entry("e_new", ["rust"])  # shares nothing
    fake = FakeClient(responses=['{"relation": "supports"}'])
    edges = link_graph(new, store, fake)
    assert edges == []
    assert fake.call_count == 0


# ---- T-G2: relation classification maps to the allowed enum only ----


@pytest.mark.unit
def test_g2_relation_within_enum(store):
    new = _entry("e_new", ["python"])
    fake = FakeClient(responses=['{"relation": "same_principle"}'])
    edges = link_graph(new, store, fake)
    assert edges[0].relation == "same_principle"


@pytest.mark.unit
def test_g2_none_relation_is_dropped(store):
    new = _entry("e_new", ["python"])
    fake = FakeClient(responses=[json.dumps({"relation": "none"})])
    edges = link_graph(new, store, fake)
    assert edges == []  # 'none' is not a real edge


@pytest.mark.unit
def test_g2_invalid_relation_is_dropped_not_crash(store):
    new = _entry("e_new", ["python"])
    fake = FakeClient(responses=['{"relation": "frenemy"}'])  # not in enum
    edges = link_graph(new, store, fake)
    assert edges == []
