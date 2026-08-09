"""Pydantic data model for Distil (SCHEMA.md §1, §2).

Two persistent objects — :class:`Profile` and a growing set of :class:`KBEntry` — plus the
nested types they own. Invariants that matter most for the product:

* ``Provenance.quote`` is **mandatory** (the format-independent faithfulness anchor);
  ``timestamp`` and ``locator`` are optional (transcripts may be untimestamped). SCHEMA §2.
* Enums are closed: an unknown ``stance``, knowledge ``type``, focus ``status``, feedback
  ``reason`` or out-of-range ``score`` is rejected, never coerced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Shared literal enums ---------------------------------------------------------------

KnowledgeType = Literal[
    "heuristic", "procedural", "declarative", "conceptual", "experiential", "opinion"
]
Stance = Literal["fact", "opinion", "personal_experience"]
Confidence = Literal["low", "medium", "high"]
Density = Literal["low", "medium", "high"]
Verdict = Literal["rich", "mixed", "little_to_extract"]
FocusStatus = Literal["active", "dormant", "archived"]
ApplicationForm = Literal["checklist", "trigger", "flashcard", "experiment", "reference"]
Relation = Literal["supports", "contradicts", "same_principle", "extends", "prerequisite_of"]
FeedbackReason = Literal[
    "relevant", "already_knew", "bad_source", "wrong_for_me", "irrelevant_now"
]


class _Model(BaseModel):
    """Base: forbid unknown fields so schema drift surfaces as a validation error."""

    model_config = ConfigDict(extra="forbid")


# ---- Profile (SCHEMA §1) ----------------------------------------------------------------


class LongTermGoal(_Model):
    id: str
    statement: str
    created_at: str


class StableProfile(_Model):
    role: str = ""
    domain: str = ""
    tools: list[str] = Field(default_factory=list)
    long_term_goals: list[LongTermGoal] = Field(default_factory=list)


class FocusItem(_Model):
    id: str
    project: str
    description: str
    active_since: str
    last_touched: str
    status: FocusStatus


class Affinities(_Model):
    topics: dict[str, float] = Field(default_factory=dict)
    knowledge_types: dict[str, float] = Field(default_factory=dict)
    application_forms: dict[str, float] = Field(default_factory=dict)


class NegativeEntry(_Model):
    weight: float = 0.0
    reasons: dict[str, int] = Field(default_factory=dict)


class Negatives(_Model):
    topics: dict[str, NegativeEntry] = Field(default_factory=dict)
    knowledge_types: dict[str, NegativeEntry] = Field(default_factory=dict)
    application_forms: dict[str, NegativeEntry] = Field(default_factory=dict)


class ProfileMeta(_Model):
    documents_processed: int = 0
    confidence: float = 0.0
    last_updated: str | None = None


class Profile(_Model):
    user_id: str
    stable: StableProfile = Field(default_factory=StableProfile)
    current_focus: list[FocusItem] = Field(default_factory=list)
    affinities: Affinities = Field(default_factory=Affinities)
    negatives: Negatives = Field(default_factory=Negatives)
    known_topics: list[str] = Field(default_factory=list)
    meta: ProfileMeta = Field(default_factory=ProfileMeta)


# ---- KBEntry (SCHEMA §2) ----------------------------------------------------------------


class Source(_Model):
    url: str | None = None
    title: str
    channel: str | None = None
    channel_url: str | None = None
    thumbnail_url: str | None = None
    metadata_provider: str | None = None
    metadata_fetched_at: str | None = None
    duration_sec: int = 0
    captured_at: str


class KnowledgeTypeShare(_Model):
    type: KnowledgeType
    share: float


class TranscriptLoss(_Model):
    level: Density
    evidence: list[str] = Field(default_factory=list)


class Triage(_Model):
    knowledge_types_present: list[KnowledgeTypeShare] = Field(default_factory=list)
    density: Density
    transcript_loss: TranscriptLoss
    verdict: Verdict


class Provenance(_Model):
    """The faithfulness anchor. ``quote`` is always present and must appear in the source."""

    quote: str
    timestamp: str | None = None
    locator: str | None = None


EntityKind = Literal["tool", "person", "organization"]


class EntityMention(_Model):
    """One named tool/person/organization mentioned in the same sentence a knowledge item was
    extracted from (Phase D). Rides along on the ``KnowledgeItem`` it was extracted alongside —
    the same extraction call, not a second transcript pass. ``kind`` is a THIRD, separate closed
    vocabulary from ``KnowledgeType``/``Stance`` — never conflate them (see prompts/extract.py).
    ``quote`` is the same short verbatim faithfulness anchor ``Provenance.quote`` enforces."""

    name: str
    kind: EntityKind
    description: str = ""
    quote: str
    timestamp: str | None = None


class KnowledgeItem(_Model):
    item_id: str
    type: KnowledgeType
    statement: str
    rationale: str | None = None  # heuristic
    scope: str | None = None  # heuristic: when it applies / doesn't
    order_index: int | None = None  # procedural
    preconditions: list[str] = Field(default_factory=list)  # procedural
    gotchas: list[str] = Field(default_factory=list)  # procedural
    stance: Stance
    speaker_confidence: Confidence = "medium"
    provenance: Provenance
    entity_mentions: list[EntityMention] = Field(default_factory=list)


class ApplicationLink(_Model):
    link_id: str
    knowledge_item_ids: list[str] = Field(default_factory=list)
    linked_goal_id: str
    application_form: ApplicationForm
    scenario: str
    novelty_flag: bool = False


class RelatedEntry(_Model):
    target: str  # entry_id or item_id
    relation: Relation


class Tags(_Model):
    topics: list[str] = Field(default_factory=list)
    knowledge_types: list[str] = Field(default_factory=list)
    application_forms: list[str] = Field(default_factory=list)


class GroundedText(_Model):
    text: str
    item_ids: list[str] = Field(default_factory=list)


class ActionStep(_Model):
    text: str
    item_ids: list[str] = Field(default_factory=list)
    application_link_ids: list[str] = Field(default_factory=list)


class ReviewQuestion(_Model):
    question: str
    item_ids: list[str] = Field(default_factory=list)


class NarrativeSummary(_Model):
    """A whole-transcript narrative account (narrative summary layer). Unlike
    ``DistilledNote``, which is built only from extracted ``KnowledgeItem`` objects and so
    inherits extraction's coverage gaps, this reads the transcript directly and is generated
    on a cheaper model tier (compression, not judgement) — see ``distil/summary.py``. Carries
    no citations: it is the readable account to read first, not the grounded structure to
    trust and check (that remains ``DistilledNote``)."""

    text: str
    chunk_count: int
    model: str
    generated_at: str


class DistilledNote(_Model):
    title: str = ""
    core_takeaway: GroundedText
    key_points: list[GroundedText] = Field(default_factory=list)
    why_it_matters: list[GroundedText] = Field(default_factory=list)
    how_to_apply: list[ActionStep] = Field(default_factory=list)
    caveats: list[GroundedText] = Field(default_factory=list)
    review_questions: list[ReviewQuestion] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    generated_from: Literal["llm", "fallback"] = "llm"


class PerLinkScore(_Model):
    link_id: str
    score: int = Field(ge=1, le=5)


class Feedback(_Model):
    score: int | None = Field(default=None, ge=1, le=5)
    reason: FeedbackReason | None = None
    per_link: list[PerLinkScore] = Field(default_factory=list)
    scored_at: str | None = None


class EntryMeta(_Model):
    created_at: str
    model_version: str = ""


class KBEntry(_Model):
    entry_id: str
    source: Source
    triage: Triage
    knowledge_items: list[KnowledgeItem] = Field(default_factory=list)
    application_links: list[ApplicationLink] = Field(default_factory=list)
    distilled_note: DistilledNote | None = None
    narrative_summary: NarrativeSummary | None = None
    related_entries: list[RelatedEntry] = Field(default_factory=list)
    tags: Tags = Field(default_factory=Tags)
    feedback: Feedback = Field(default_factory=Feedback)
    meta: EntryMeta


# ---- Concept (Phase 15.1 — canonicalize engine; OKF Phase 3a design report §3) ----------


class ConceptMember(_Model):
    """One knowledge item's membership in a concept, copied at match/new-creation time."""

    entry_id: str
    item_id: str
    quote: str
    timestamp: str | None = None


class ConceptClaim(_Model):
    """A synthesized sentence/paragraph and the members it's grounded in — same shape as
    ``GroundedText``, reused at concept scope (Phase 15.2 design report §3, §4): the LLM
    writes text, code enforces every claim traces to real membership before it's rendered."""

    text: str
    item_ids: list[str] = Field(default_factory=list)


ConceptRelation = Literal["contrasts_with", "builds_on", "related"]


class ConceptEdge(_Model):
    """One typed, directional link from this concept to another (Phase 16, OKF Phase 3b design
    report §9 item 4). Computed independently whenever the source concept is (re-)synthesized —
    same "from this side's perspective" shape ``RelatedEntry`` already has at entry granularity."""

    target_concept_id: str
    relation: ConceptRelation


class Concept(_Model):
    """A canonical idea spanning one or more videos. ``concept_id`` == the slug (OKF's "path
    is identity"). No feedback/application-link data by construction — see report §3."""

    concept_id: str
    title: str
    description: str
    members: list[ConceptMember] = Field(default_factory=list)
    claims: list[ConceptClaim] = Field(default_factory=list)
    edges: list[ConceptEdge] = Field(default_factory=list)
    created_at: str
    updated_at: str
    body_model_version: str = ""
    pending_synthesis: bool = False


# ---- Entity (Phase D — entities layer; canonicalize.py mirrors the Concept shape) -------


class EntityMember(_Model):
    """One knowledge item's mention of an entity, copied at match/new-creation time — the exact
    shape of ``ConceptMember`` (design intentionally reused, see canonicalize.py)."""

    entry_id: str
    item_id: str
    quote: str
    timestamp: str | None = None


class EntityClaim(_Model):
    """A synthesized sentence/paragraph about an entity and the members it's grounded in — same
    shape as ``ConceptClaim``, reused at entity scope."""

    text: str
    item_ids: list[str] = Field(default_factory=list)


class Entity(_Model):
    """A canonical tool/person/organization spanning one or more videos. ``entity_id`` == the
    slug (OKF's "path is identity"), mirroring ``Concept``. ``kind`` is fixed at creation from
    the first mention's ``EntityMention.kind`` and never changes — canonicalize only merges
    mentions of the same ``kind`` into an existing entity (see ``Store.find_entity_candidates``)."""

    entity_id: str
    kind: EntityKind
    title: str
    description: str
    members: list[EntityMember] = Field(default_factory=list)
    claims: list[EntityClaim] = Field(default_factory=list)
    created_at: str
    updated_at: str
    body_model_version: str = ""
    pending_synthesis: bool = False
