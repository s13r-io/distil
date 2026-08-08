"""Phase 15.3 — canonicalize eval against a real model + real embedder. Design report §6, §8.

The validation gate for the hybrid matching engine: unit tests (``tests/unit/test_canonicalize.py``)
prove the glue is correct against a ``FakeClient``, but only a real model's *judgment* can show
whether embedding similarity + LLM classification actually clusters real prose the way §1's
"traditional RAG vs agentic RAG" example expects. Marked ``eval`` (gated by ``ANTHROPIC_API_KEY`` +
``DISTIL_MODEL``; needs the ``embed-local`` extra for a real embedder, skipped if unavailable —
mirrors ``tests/eval/test_query_eval.py``'s skip pattern).

T-CANON-EVAL1 (merge precision/recall) and T-CANON-EVAL2 (synthesis faithfulness under real
model output) assert *properties*, not exact strings, per ``docs/TESTING.md``'s eval-suite style.
"""

from __future__ import annotations

import pytest

from distil.canonicalize import canonicalize_entry
from distil.llm import AnthropicClient
from distil.models import Concept, ConceptMember, KBEntry
from distil.store import Store
from distil.synthesize_concept import synthesize_concept

# Three paraphrases of the same idea (naive/traditional RAG: retrieve once, then generate, no
# planning loop), in genuinely different words -- the property the merge gate is checking is
# that embedding similarity + LLM judgment recognize these as one concept despite the lexical
# variation, not that they share vocabulary.
_TRADITIONAL_RAG_PARAPHRASES = [
    (
        "Traditional RAG retrieves relevant documents from a vector database and then feeds "
        "them to the language model to generate an answer, with no additional reasoning step "
        "in between."
    ),
    (
        "Naive RAG works by first pulling matching context chunks from your document store, "
        "then handing them directly to the model to produce a response -- there's no planning "
        "or iteration involved at all."
    ),
    (
        "In basic RAG pipelines you search for similar text chunks, insert them into the "
        "prompt, and let the language model generate the final answer in a single pass."
    ),
]

# Lexically adjacent (shares "RAG", "retrieval", "generate") but a genuinely distinct idea: an
# extra planning/decision loop around retrieval, not a single retrieve-then-generate pass.
_AGENTIC_RAG_DISTINCT = (
    "Agentic RAG adds a planning loop before retrieval: the agent decides what to search for, "
    "evaluates whether the retrieved documents are sufficient, and can issue additional "
    "retrieval calls before finally generating an answer."
)


def _make_embedder():
    try:
        from distil.embed import LocalEmbedder

        return LocalEmbedder()
    except Exception:  # pragma: no cover
        pytest.skip("local embedder not available")


def _entry(entry_id: str, title: str, item_id: str, statement: str) -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
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
                    "provenance": {"quote": statement[:60]},
                }
            ],
            "tags": {"topics": ["rag"], "knowledge_types": ["conceptual"], "application_forms": []},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _concept_id_for(store: Store, entry_id: str) -> str | None:
    for concept in store.list_concepts():
        if any(m.entry_id == entry_id for m in concept.members):
            return concept.concept_id
    return None


@pytest.mark.eval
def test_canon_eval1_paraphrases_merge_and_distinct_idea_stays_out(tmp_path):
    store = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    embedder = _make_embedder()
    client = AnthropicClient()

    paraphrase_entry_ids = []
    for i, statement in enumerate(_TRADITIONAL_RAG_PARAPHRASES):
        entry = _entry(f"e_para{i}", f"RAG explainer {i}", f"k_{i}", statement)
        store.file_entry(entry, embedder=embedder)
        canonicalize_entry(entry, store, client)
        paraphrase_entry_ids.append(entry.entry_id)

    distinct_entry = _entry("e_agentic", "Agentic RAG explained", "k_agentic", _AGENTIC_RAG_DISTINCT)
    store.file_entry(distinct_entry, embedder=embedder)
    canonicalize_entry(distinct_entry, store, client)

    paraphrase_concept_ids = {_concept_id_for(store, eid) for eid in paraphrase_entry_ids}
    assert None not in paraphrase_concept_ids, "every paraphrase should be concept-worthy"
    assert len(paraphrase_concept_ids) == 1, (
        f"the 3 paraphrases should land in ONE concept, got {paraphrase_concept_ids}"
    )

    distinct_concept_id = _concept_id_for(store, distinct_entry.entry_id)
    assert distinct_concept_id != next(iter(paraphrase_concept_ids)), (
        "agentic RAG is a genuinely distinct idea and must not merge into the traditional-RAG "
        f"concept (both resolved to {distinct_concept_id!r})"
    )


@pytest.mark.eval
def test_canon_eval2_synthesis_faithfulness_under_real_model(tmp_path):
    store = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    client = AnthropicClient()

    entries = [
        _entry(f"e_syn{i}", f"RAG explainer {i}", f"k_{i}", statement)
        for i, statement in enumerate(_TRADITIONAL_RAG_PARAPHRASES)
    ]
    for entry in entries:
        store.file_entry(entry)

    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve once, then generate an answer, with no planning loop.",
        members=[
            ConceptMember(
                entry_id=entry.entry_id,
                item_id=entry.knowledge_items[0].item_id,
                quote=entry.knowledge_items[0].provenance.quote,
                timestamp=None,
            )
            for entry in entries
        ],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )

    synthesized = synthesize_concept(concept, store, client)

    assert synthesized.claims, "real model output should survive cleaning with >=1 kept claim"
    valid_item_ids = {m.item_id for m in synthesized.members}
    for claim in synthesized.claims:
        assert claim.item_ids, "every kept claim must cite at least one item_id"
        assert all(item_id in valid_item_ids for item_id in claim.item_ids), (
            f"claim cites an item_id outside the concept's members: {claim.item_ids}"
        )
