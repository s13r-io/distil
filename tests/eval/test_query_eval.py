"""Phase 10.7 — read-layer eval. Test T-Q7.

On the query KB fixture: answerable questions return correct sources; no-note questions abstain
100% of the time (zero answers from outside knowledge). Marked ``eval`` (gated by API key).

Uses the real configured embedder; if a local embedder isn't installed the test skips rather
than fail (the guarantee is exercised hermetically by tests/unit/test_query.py::T-Q2/T-Q3).
"""

import json
from pathlib import Path

import pytest

from distil.llm import AnthropicClient
from distil.models import KBEntry
from distil.query import ask
from distil.store import Store

FIX = Path(__file__).parent.parent / "fixtures" / "query_kb"


def _make_embedder():
    try:
        from distil.embed import LocalEmbedder

        return LocalEmbedder()
    except Exception:  # pragma: no cover
        pytest.skip("local embedder not available")


def _entry(store, entry_id, items):
    store.file_entry(KBEntry.model_validate({
        "entry_id": entry_id,
        "source": {"title": entry_id, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "declarative", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [
            {"item_id": iid, "type": "declarative", "statement": stmt, "stance": "fact",
             "provenance": {"quote": stmt[:20].lower()}}
            for iid, stmt in items
        ],
        "tags": {"topics": [], "knowledge_types": ["declarative"], "application_forms": []},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    }), embedder=_EMB)


_EMB = None


@pytest.fixture
def kb(tmp_path):
    global _EMB
    _EMB = _make_embedder()
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    _entry(s, "e_test", [
        ("k_t1", "Write unit tests before the implementation to clarify the behavior."),
        ("k_t2", "Keep tests fast and isolated so they run on every change."),
    ])
    _entry(s, "e_k8s", [
        ("k_k1", "Kubernetes pods share a network namespace and can reach each other on localhost."),
    ])
    return s


@pytest.mark.eval
def test_q7_answerable_questions_return_correct_sources(kb):
    spec = json.loads((FIX / "questions.json").read_text())
    client = AnthropicClient()
    for q in spec["answerable"]:
        result = ask(q["question"], kb, _EMB, client, threshold=0.2, top_k=4)
        assert not result.abstained, f"should answer: {q['question']}"
        joined = " ".join(s.quote for s in result.sources).lower()
        assert q["expected_item_substring"] in joined


@pytest.mark.eval
def test_q7_no_note_questions_abstain_100pct(kb):
    spec = json.loads((FIX / "questions.json").read_text())
    client = AnthropicClient()
    for question in spec["no_notes"]:
        result = ask(question, kb, _EMB, client, threshold=0.35, top_k=4)
        assert result.abstained, f"must abstain (no outside knowledge): {question}"
        assert result.answer is None
