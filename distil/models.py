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
    related_entries: list[RelatedEntry] = Field(default_factory=list)
    tags: Tags = Field(default_factory=Tags)
    feedback: Feedback = Field(default_factory=Feedback)
    meta: EntryMeta
