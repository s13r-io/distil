"""Phase 15.2 — synthesize_concept.py: concept-page synthesis + code-rendered citations.

Tests T-SYN1-4 (design report §4, §8).
"""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Concept, ConceptClaim, ConceptMember, KBEntry
from distil.store import Store
from distil.synthesize_concept import render_claim, synthesize_concept


def _entry(entry_id, item_id, statement, quote, timestamp=None) -> KBEntry:
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
                    "provenance": {"quote": quote, "timestamp": timestamp},
                }
            ],
            "tags": {
                "topics": ["rag"],
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
    return FakeEmbedder(dim=16)


@pytest.fixture
def concept(store, embedder):
    e1 = _entry(
        "e_01", "k_01", "Traditional RAG retrieves then generates.", "retrieve then generate", "0:01:04"
    )
    e2 = _entry("e_02", "k_02", "Naive RAG has no planning step.", "no planning step", "0:02:10")
    store.file_entry(e1, embedder=embedder)
    store.file_entry(e2, embedder=embedder)
    return Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve documents, then generate an answer.",
        members=[
            ConceptMember(
                entry_id="e_01", item_id="k_01", quote="retrieve then generate", timestamp="0:01:04"
            ),
            ConceptMember(entry_id="e_02", item_id="k_02", quote="no planning step", timestamp="0:02:10"),
        ],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )


# ---- T-SYN1: valid claims JSON parses into cleaned ConceptClaims ----


@pytest.mark.unit
def test_syn1_valid_claims_parse(store, concept):
    raw = json.dumps(
        [
            {
                "text": "Traditional RAG retrieves then generates with no planning loop.",
                "item_ids": ["k_01", "k_02"],
            },
        ]
    )
    result = synthesize_concept(concept, store, FakeClient([raw]))
    assert len(result.claims) == 1
    assert result.claims[0].item_ids == ["k_01", "k_02"]
    assert result.pending_synthesis is False


# ---- T-SYN2: a claim citing an unknown item_id is dropped, not trusted ----


@pytest.mark.unit
def test_syn2_claim_citing_unknown_item_is_dropped(store, concept):
    raw = json.dumps(
        [
            {"text": "Valid claim.", "item_ids": ["k_01"]},
            {"text": "Hallucinated claim.", "item_ids": ["k_99"]},
        ]
    )
    result = synthesize_concept(concept, store, FakeClient([raw]))
    assert len(result.claims) == 1
    assert result.claims[0].text == "Valid claim."


@pytest.mark.unit
def test_syn2_mixed_valid_and_unknown_ids_drops_whole_claim(store, concept):
    # design report §4 step 3: drop claims whose item_ids don't ALL resolve, not just the
    # unknown id — a mixed valid/invalid citation is not trusted at concept granularity. With
    # a second, fully-valid claim also present, cleaning yields just that one (no fallback).
    raw = json.dumps(
        [
            {"text": "Mixed claim.", "item_ids": ["k_01", "k_99"]},
            {"text": "Clean claim.", "item_ids": ["k_02"]},
        ]
    )
    result = synthesize_concept(concept, store, FakeClient([raw]))
    assert len(result.claims) == 1
    assert result.claims[0].text == "Clean claim."


# ---- T-SYN3: malformed output falls back to a deterministic one-liner, never raises ----


@pytest.mark.unit
def test_syn3_malformed_output_falls_back(store, concept):
    result = synthesize_concept(concept, store, FakeClient(["not json"]))
    assert len(result.claims) == 1
    assert result.claims[0].item_ids == ["k_01", "k_02"]
    assert "Traditional RAG retrieves then generates." in result.claims[0].text
    assert "Retrieve documents, then generate an answer." in result.claims[0].text


@pytest.mark.unit
def test_syn3_empty_claims_after_cleaning_falls_back(store, concept):
    # Every claim cites an unknown item_id -> cleaned list is empty -> deterministic fallback.
    raw = json.dumps([{"text": "Hallucinated.", "item_ids": ["k_99"]}])
    result = synthesize_concept(concept, store, FakeClient([raw]))
    assert len(result.claims) == 1
    assert result.claims[0].item_ids == ["k_01", "k_02"]


# ---- T-SYN4: rendered citation is code-derived from validated data, never model text ----


@pytest.mark.unit
def test_syn4_citation_rendering_matches_code_derived_data():
    claim = ConceptClaim(text="Traditional RAG retrieves then generates.", item_ids=["k_01"])
    citations = {"k_01": ("why-ai-abandoned-rag", "0:01:04")}
    rendered = render_claim(claim, citations)
    assert rendered == "Traditional RAG retrieves then generates. (why-ai-abandoned-rag, 0:01:04)."


@pytest.mark.unit
def test_syn4_multiple_citations_are_comma_separated_in_one_parenthetical():
    claim = ConceptClaim(text="Two videos agree.", item_ids=["k_01", "k_02"])
    citations = {
        "k_01": ("why-ai-abandoned-rag", "0:01:04"),
        "k_02": ("rag-20-agentic-rag", "0:02:10"),
    }
    rendered = render_claim(claim, citations)
    assert rendered == "Two videos agree. (why-ai-abandoned-rag, 0:01:04, rag-20-agentic-rag, 0:02:10)."


@pytest.mark.unit
def test_syn4_citation_without_timestamp_omits_it():
    claim = ConceptClaim(text="No timestamp here.", item_ids=["k_01"])
    citations = {"k_01": ("some-slug", None)}
    rendered = render_claim(claim, citations)
    assert rendered == "No timestamp here. (some-slug)."
