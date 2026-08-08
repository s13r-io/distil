"""Phase 1.1 — Pydantic models (SCHEMA.md §1, §2). Tests T-M1..M4."""

import pytest
from pydantic import ValidationError

from distil.models import (
    ApplicationLink,
    Concept,
    ConceptClaim,
    ConceptMember,
    Entity,
    EntityClaim,
    EntityMember,
    EntityMention,
    Feedback,
    KBEntry,
    KnowledgeItem,
    Profile,
    Provenance,
)

# ---- T-M1: Profile validates; rejects bad status enum ----


@pytest.mark.unit
def test_profile_minimal_validates():
    p = Profile(user_id="u1")
    assert p.user_id == "u1"
    assert p.meta.documents_processed == 0
    assert p.meta.confidence == 0.0


@pytest.mark.unit
def test_profile_rejects_bad_focus_status():
    with pytest.raises(ValidationError):
        Profile(
            user_id="u1",
            current_focus=[
                {
                    "id": "f_01",
                    "project": "x",
                    "description": "d",
                    "active_since": "2026-01-01T00:00:00",
                    "last_touched": "2026-01-01T00:00:00",
                    "status": "not_a_status",
                }
            ],
        )


@pytest.mark.unit
def test_profile_accepts_valid_focus_status():
    p = Profile(
        user_id="u1",
        current_focus=[
            {
                "id": "f_01",
                "project": "x",
                "description": "d",
                "active_since": "2026-01-01T00:00:00",
                "last_touched": "2026-01-01T00:00:00",
                "status": "active",
            }
        ],
    )
    assert p.current_focus[0].status == "active"


# ---- T-M2: KnowledgeItem requires provenance; quote mandatory, timestamp may be null ----


@pytest.mark.unit
def test_knowledge_item_requires_provenance():
    with pytest.raises(ValidationError):
        KnowledgeItem(item_id="k_01", type="heuristic", statement="s", stance="fact")


@pytest.mark.unit
def test_provenance_quote_mandatory():
    with pytest.raises(ValidationError):
        Provenance(timestamp="00:01:00")  # no quote


@pytest.mark.unit
def test_provenance_timestamp_optional():
    prov = Provenance(quote="keep functions small")
    assert prov.timestamp is None
    assert prov.locator is None


@pytest.mark.unit
def test_knowledge_item_valid_with_quote_only():
    item = KnowledgeItem(
        item_id="k_01",
        type="heuristic",
        statement="Keep functions small.",
        stance="opinion",
        provenance={"quote": "keep functions small", "locator": "seg:3"},
    )
    assert item.provenance.quote == "keep functions small"
    assert item.provenance.timestamp is None


# ---- T-M3: stance enum enforced ----


@pytest.mark.unit
def test_stance_enum_enforced():
    with pytest.raises(ValidationError):
        KnowledgeItem(
            item_id="k_01",
            type="heuristic",
            statement="s",
            stance="rumour",
            provenance={"quote": "q"},
        )


@pytest.mark.unit
def test_knowledge_type_enum_enforced():
    with pytest.raises(ValidationError):
        KnowledgeItem(
            item_id="k_01",
            type="gossip",
            statement="s",
            stance="fact",
            provenance={"quote": "q"},
        )


# ---- T-M4: round-trip serialize -> deserialize is lossless ----


@pytest.mark.unit
def test_kbentry_round_trip_lossless():
    entry = KBEntry(
        entry_id="e_01",
        source={"title": "A talk", "captured_at": "2026-06-15T00:00:00"},
        triage={
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        knowledge_items=[
            KnowledgeItem(
                item_id="k_01",
                type="heuristic",
                statement="Keep functions small.",
                rationale="Easier to test.",
                scope="When writing library code.",
                stance="opinion",
                speaker_confidence="high",
                provenance={"quote": "keep functions small", "timestamp": "00:12:30"},
            )
        ],
        application_links=[
            ApplicationLink(
                link_id="a_01",
                knowledge_item_ids=["k_01"],
                linked_goal_id="g_01",
                application_form="checklist",
                scenario="Refactor the auth module.",
            )
        ],
        feedback=Feedback(),
        meta={"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    )
    dumped = entry.model_dump_json()
    restored = KBEntry.model_validate_json(dumped)
    assert restored == entry
    assert restored.distilled_note is None


@pytest.mark.unit
def test_kbentry_round_trip_with_distilled_note():
    entry = KBEntry(
        entry_id="e_01",
        source={"title": "A talk", "captured_at": "2026-06-15T00:00:00"},
        triage={
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        knowledge_items=[
            KnowledgeItem(
                item_id="k_01",
                type="heuristic",
                statement="Keep functions small.",
                stance="opinion",
                provenance={"quote": "keep functions small"},
            )
        ],
        distilled_note={
            "title": "Small functions",
            "core_takeaway": {
                "text": "Small functions are easier to reason about.",
                "item_ids": ["k_01"],
            },
            "topics": ["function_design"],
        },
        meta={"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    )
    restored = KBEntry.model_validate_json(entry.model_dump_json())
    assert restored == entry
    assert restored.distilled_note.core_takeaway.item_ids == ["k_01"]


@pytest.mark.unit
def test_feedback_defaults_to_unscored():
    fb = Feedback()
    assert fb.score is None
    assert fb.reason is None
    assert fb.scored_at is None


@pytest.mark.unit
def test_feedback_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        Feedback(score=6)


@pytest.mark.unit
def test_feedback_rejects_bad_reason():
    with pytest.raises(ValidationError):
        Feedback(score=2, reason="meh")


# ---- Concept / ConceptMember (Phase 15.1 — canonicalize engine, design report §3) --------


@pytest.mark.unit
def test_concept_member_requires_quote():
    with pytest.raises(ValidationError):
        ConceptMember(entry_id="e_01", item_id="k_01")


@pytest.mark.unit
def test_concept_member_timestamp_optional():
    member = ConceptMember(entry_id="e_01", item_id="k_01", quote="q")
    assert member.timestamp is None


@pytest.mark.unit
def test_concept_round_trip_lossless():
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve then generate.",
        members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q", timestamp="00:01:00")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    restored = Concept.model_validate_json(concept.model_dump_json())
    assert restored == concept


@pytest.mark.unit
def test_concept_claim_validates_like_grounded_text():
    claim = ConceptClaim(text="Traditional RAG retrieves then generates.", item_ids=["k_01"])
    assert claim.text == "Traditional RAG retrieves then generates."
    assert claim.item_ids == ["k_01"]


@pytest.mark.unit
def test_concept_claim_item_ids_default_empty():
    claim = ConceptClaim(text="x")
    assert claim.item_ids == []


@pytest.mark.unit
def test_concept_has_claims_and_pending_synthesis_fields():
    # Phase 15.2 (design report §3, §6) adds the synthesized body + the synthesis-capping flag.
    assert "claims" in Concept.model_fields
    assert "pending_synthesis" in Concept.model_fields
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve then generate.",
        members=[ConceptMember(entry_id="e_01", item_id="k_01", quote="q")],
        claims=[ConceptClaim(text="Retrieve then generate.", item_ids=["k_01"])],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    assert concept.claims[0].text == "Retrieve then generate."
    assert concept.pending_synthesis is False


# ---- Entity / EntityMember / EntityMention (Phase D — entities layer) --------------------


@pytest.mark.unit
def test_entity_kind_enum_enforced():
    with pytest.raises(ValidationError):
        EntityMention(name="React", kind="framework", quote="q")


@pytest.mark.unit
def test_entity_mention_valid_kinds_accepted():
    for kind in ("tool", "person", "organization"):
        mention = EntityMention(name="X", kind=kind, quote="q")
        assert mention.kind == kind


@pytest.mark.unit
def test_knowledge_item_entity_mentions_default_empty():
    item = KnowledgeItem(
        item_id="k_01", type="conceptual", statement="s", stance="fact",
        provenance=Provenance(quote="q"),
    )
    assert item.entity_mentions == []


@pytest.mark.unit
def test_knowledge_item_carries_entity_mentions():
    item = KnowledgeItem(
        item_id="k_01", type="conceptual", statement="s", stance="fact",
        provenance=Provenance(quote="q"),
        entity_mentions=[EntityMention(name="React", kind="tool", quote="react")],
    )
    assert item.entity_mentions[0].name == "React"


@pytest.mark.unit
def test_entity_member_requires_quote():
    with pytest.raises(ValidationError):
        EntityMember(entry_id="e_01", item_id="k_01")


@pytest.mark.unit
def test_entity_member_timestamp_optional():
    member = EntityMember(entry_id="e_01", item_id="k_01", quote="q")
    assert member.timestamp is None


@pytest.mark.unit
def test_entity_round_trip_lossless():
    entity = Entity(
        entity_id="react",
        kind="tool",
        title="React",
        description="A JS UI library.",
        members=[EntityMember(entry_id="e_01", item_id="k_01", quote="q", timestamp="00:01:00")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    restored = Entity.model_validate_json(entity.model_dump_json())
    assert restored == entity


@pytest.mark.unit
def test_entity_kind_enum_enforced_at_entity_level():
    with pytest.raises(ValidationError):
        Entity(
            entity_id="react", kind="framework", title="React", description="d",
            created_at="t", updated_at="t",
        )


@pytest.mark.unit
def test_entity_claim_validates_like_concept_claim():
    claim = EntityClaim(text="React uses a virtual DOM.", item_ids=["k_01"])
    assert claim.text == "React uses a virtual DOM."
    assert claim.item_ids == ["k_01"]


@pytest.mark.unit
def test_entity_claim_item_ids_default_empty():
    assert EntityClaim(text="x").item_ids == []


@pytest.mark.unit
def test_entity_has_claims_and_pending_synthesis_fields():
    assert "claims" in Entity.model_fields
    assert "pending_synthesis" in Entity.model_fields
    entity = Entity(
        entity_id="react", kind="tool", title="React", description="d",
        members=[EntityMember(entry_id="e_01", item_id="k_01", quote="q")],
        claims=[EntityClaim(text="React uses a virtual DOM.", item_ids=["k_01"])],
        created_at="2026-06-15T00:00:00", updated_at="2026-06-15T00:00:00",
    )
    assert entity.claims[0].text == "React uses a virtual DOM."
    assert entity.pending_synthesis is False
