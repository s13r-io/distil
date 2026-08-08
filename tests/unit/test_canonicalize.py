"""Phase 15.1 — canonicalize.py: match/new/reject matching engine.

Tests T-CANON1-7, T-CANON9 (design report §8; T-CANON8, synthesis capping, is 15.2 scope).
"""

import json

import pytest

from distil.canonicalize import canonicalize_entry
from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Concept, ConceptMember, KBEntry
from distil.store import Store

_AGENTIC_RAG = "Agentic RAG adds a planning loop before retrieval"
_TRADITIONAL_RAG = "Traditional RAG retrieves documents then generates an answer"


def _item(item_id, statement, *, quote="q", stance="fact", type_="conceptual"):
    return {
        "item_id": item_id,
        "type": type_,
        "statement": statement,
        "stance": stance,
        "provenance": {"quote": quote},
    }


def _entry(entry_id, items, topics=("rag",)) -> KBEntry:
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
            "knowledge_items": items,
            "tags": {
                "topics": list(topics),
                "knowledge_types": ["conceptual"],
                "application_forms": [],
            },
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


def _vector_for(store: Store, entry_id: str) -> list[float]:
    return next(v for _iid, eid, v in store.iter_item_vectors() if eid == entry_id)


# ---- T-CANON1: near-match becomes a member of the existing concept, no new concept ----


@pytest.mark.unit
def test_canon1_near_match_becomes_member_no_new_concept(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _TRADITIONAL_RAG)])
    store.file_entry(e1, embedder=embedder)
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve then generate.",
        members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    store.save_concept(concept)

    e2 = _entry("e_02", [_item("k_01", _TRADITIONAL_RAG)])
    store.file_entry(e2, embedder=embedder)
    fake = FakeClient(
        responses=[
            json.dumps([{"item_id": "k_01", "decision": "match", "concept_id": "traditional-rag"}])
        ]
    )

    touched = canonicalize_entry(e2, store, fake)

    assert [c.concept_id for c in touched] == ["traditional-rag"]
    assert len(store.list_concepts()) == 1
    saved = store.load_concept("traditional-rag")
    assert {m.entry_id for m in saved.members} == {"e_01", "e_02"}


# ---- T-CANON2: empty candidate pool + "new" creates a fresh concept ----


@pytest.mark.unit
def test_canon2_empty_pool_creates_new_concept(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(
        responses=[
            json.dumps(
                [{"item_id": "k_01", "decision": "new", "title": "Agentic RAG",
                  "description": "RAG with a planning loop."}]
            )
        ]
    )

    touched = canonicalize_entry(e1, store, fake)

    assert len(touched) == 1
    concepts = store.list_concepts()
    assert len(concepts) == 1
    assert concepts[0].concept_id == "agentic-rag"
    assert concepts[0].members == [
        ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp=None)
    ]


# ---- T-CANON3: re-canonicalizing the same entry is idempotent (exact members equality) ----


@pytest.mark.unit
def test_canon3_refiling_same_entry_is_idempotent(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder)
    response = json.dumps(
        [{"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d"}]
    )

    canonicalize_entry(e1, store, FakeClient(responses=[response]))
    first_members = store.load_concept("agentic-rag").members

    canonicalize_entry(e1, store, FakeClient(responses=[response]))
    second_members = store.load_concept("agentic-rag").members

    assert first_members == second_members
    assert len(store.list_concepts()) == 1


# ---- T-CANON4: an out-of-enum decision is treated as reject, never raises ----


@pytest.mark.unit
def test_canon4_invalid_decision_is_dropped_not_crash(store, embedder):
    e1 = _entry("e_01", [_item("k_01", "Some fact about the speaker's morning routine")])
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(responses=[json.dumps([{"item_id": "k_01", "decision": "frenemy"}])])

    touched = canonicalize_entry(e1, store, fake)

    assert touched == []
    assert store.list_concepts() == []


# ---- T-CANON5: a match to a concept_id that was never offered is dropped, not trusted ----


@pytest.mark.unit
def test_canon5_untrusted_concept_id_is_dropped(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder)
    # No concepts exist yet, so k_01's candidate pool is empty; the model hallucinates anyway.
    fake = FakeClient(
        responses=[
            json.dumps([{"item_id": "k_01", "decision": "match", "concept_id": "ghost-concept"}])
        ]
    )

    touched = canonicalize_entry(e1, store, fake)

    assert touched == []
    assert store.list_concepts() == []


# ---- T-CANON6: candidate pool sent to the LLM is bounded at MAX_CONCEPT_CANDIDATES ----


@pytest.mark.unit
def test_canon6_candidate_pool_is_bounded_in_prompt(store, embedder, monkeypatch):
    monkeypatch.delenv("DISTIL_CONCEPT_MAX_CANDIDATES", raising=False)
    monkeypatch.delenv("DISTIL_CONCEPT_SIM_FLOOR", raising=False)
    for i in range(7):
        entry = _entry(f"e_c{i}", [_item("k_01", _TRADITIONAL_RAG)])
        store.file_entry(entry, embedder=embedder)
        store.save_concept(
            Concept(
                concept_id=f"concept-{i}",
                title=f"Concept {i}",
                description="d",
                members=[ConceptMember(entry_id=f"e_c{i}", item_id="k_01", quote="q", timestamp=None)],
                created_at="t",
                updated_at="t",
            )
        )

    new_entry = _entry("e_new", [_item("k_01", _TRADITIONAL_RAG)])
    store.file_entry(new_entry, embedder=embedder)
    fake = FakeClient(responses=[json.dumps([{"item_id": "k_01", "decision": "reject"}])])

    canonicalize_entry(new_entry, store, fake)

    prompt = fake.calls[0].prompt
    items_block = prompt.split("ITEMS AND THEIR CANDIDATE CONCEPTS:\n", 1)[1].split(
        "\n\nFor each item, return exactly one decision:", 1
    )[0]
    payload = json.loads(items_block)
    assert len(payload[0]["candidates"]) == 5  # 7 qualifying candidates capped at the default 5


# ---- T-CANON7: same-batch "new" proposals with the same normalized title merge ----


@pytest.mark.unit
def test_canon7_same_batch_new_title_collision_merges(store, embedder):
    e1 = _entry(
        "e_01",
        [
            _item("k_01", "Agentic RAG adds a planning loop before retrieval", quote="q1"),
            _item("k_02", "Agentic RAG lets the model decide when to retrieve", quote="q2"),
        ],
    )
    store.file_entry(e1, embedder=embedder)
    fake = FakeClient(
        responses=[
            json.dumps(
                [
                    {"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d1"},
                    {"item_id": "k_02", "decision": "new", "title": "agentic rag!", "description": "d2"},
                ]
            )
        ]
    )

    touched = canonicalize_entry(e1, store, fake)

    assert len(touched) == 1
    concepts = store.list_concepts()
    assert len(concepts) == 1
    assert {m.item_id for m in concepts[0].members} == {"k_01", "k_02"}


# ---- T-CANON9: delete_entry cascades retraction, cleaning up zero-member concepts ----


@pytest.mark.unit
def test_canon9_delete_entry_cascades_concept_retraction(store, embedder):
    e1 = _entry("e_01", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e1, embedder=embedder)
    canonicalize_entry(
        e1,
        store,
        FakeClient(
            responses=[
                json.dumps([{"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d"}])
            ]
        ),
    )
    concept_id = store.list_concepts()[0].concept_id

    # Sole source: deleting it removes the concept entirely.
    store.delete_entry("e_01")
    assert store.load_concept(concept_id) is None
    assert store.list_concepts() == []

    # Two sources: deleting one only retracts that membership, leaving the other.
    e2 = _entry("e_02", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e2, embedder=embedder)
    canonicalize_entry(
        e2,
        store,
        FakeClient(
            responses=[
                json.dumps([{"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d"}])
            ]
        ),
    )
    cid = store.list_concepts()[0].concept_id

    e3 = _entry("e_03", [_item("k_01", _AGENTIC_RAG)])
    store.file_entry(e3, embedder=embedder)
    canonicalize_entry(
        e3,
        store,
        FakeClient(
            responses=[json.dumps([{"item_id": "k_01", "decision": "match", "concept_id": cid}])]
        ),
    )
    assert len(store.load_concept(cid).members) == 2

    store.delete_entry("e_02")
    remaining = store.load_concept(cid)
    assert remaining is not None
    assert {m.entry_id for m in remaining.members} == {"e_03"}
