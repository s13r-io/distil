"""Read layer — retrieve → relevance gate → grounded synthesis. ARCHITECTURE.md §9.

The gate is the no-hallucination guarantee in code: synthesis is **never** invoked unless
retrieval clears the threshold (T-Q2), so the system either grounds an answer in the user's
notes or honestly says it has none — it cannot answer from the model's outside knowledge.
Grounding is enforced after synthesis too: any citation to an item outside the retrieved set
is stripped out and reported, never presented as a real source (T-Q3).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .embed import Embedder
from .llm import LLMClient
from .models import ConceptClaim, EntityClaim
from .prompts.synthesize import SYSTEM, build_synthesis_prompt
from .store import Store

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CITATION = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# Ranking weights for similarity × feedback × recency.
_RECENCY_HALF_LIFE_DAYS = 180.0

# Depth of concept substance sent to synthesis (Phase C). "claims" sends each claim's own
# sentence only (pre-Phase-C behavior); "full" additionally sends the real member quotes each
# claim cites, i.e. the same evidence a reader would see after following the link to the concept
# page — see the module docstring addendum below and DEPLOY-relevant CLAUDE.md entry for the
# measured token-cost tradeoff behind the default.
CONCEPT_NOTE_DEPTH_DEFAULT = "claims"
_CONCEPT_NOTE_DEPTHS = {"claims", "full"}


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0 if either is zero-length)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RetrievedItem:
    item_id: str
    entry_id: str
    statement: str
    quote: str
    timestamp: str | None
    similarity: float
    score: float  # composite rank score (similarity × feedback × recency)
    entry_title: str = ""
    context: str = ""


@dataclass
class RetrievedConcept:
    concept_id: str
    title: str
    description: str
    similarity: float
    claims: list[ConceptClaim] = field(default_factory=list)


@dataclass
class RetrievedEntity:
    entity_id: str
    kind: str
    title: str
    description: str
    similarity: float
    claims: list[EntityClaim] = field(default_factory=list)


@dataclass
class Source:
    item_id: str
    entry_id: str
    quote: str
    timestamp: str | None
    entry_title: str = ""


@dataclass
class ConceptRef:
    """A concept page that cleared the gate and fed this answer — enough to link to
    ``/concepts/{concept_id}`` (Phase C). Carries no claim/member data: the concept detail page
    already renders those, so this is just "what contributed," not a duplicate of the page."""

    concept_id: str
    title: str


@dataclass
class AskResult:
    abstained: bool
    message: str = ""
    answer: str | None = None
    sources: list[Source] = field(default_factory=list)
    cited_item_ids: list[str] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    conflict: str | None = None
    concepts: list[ConceptRef] = field(default_factory=list)


def retrieve(
    question: str, store: Store, embedder: Embedder, *, top_k: int = 6
) -> list[RetrievedItem]:
    qvec = embedder.embed(question)
    # entry-level metadata for feedback score + recency.
    entry_meta = {r.entry_id: r for r in store.list_entries()}
    now = datetime.now(timezone.utc)

    scored: list[RetrievedItem] = []
    for item_id, entry_id, vec in store.iter_item_vectors():
        sim = cosine(qvec, vec)
        meta = entry_meta.get(entry_id)
        feedback_mult = _feedback_multiplier(meta.score if meta else None)
        recency_mult = _recency_multiplier(meta.created_at if meta else None, now)
        composite = sim * feedback_mult * recency_mult
        loaded = _load_item(store, entry_id, item_id)
        if loaded is None:
            continue
        item, context = loaded
        scored.append(RetrievedItem(
            item_id=item_id, entry_id=entry_id, statement=item.statement,
            quote=item.provenance.quote, timestamp=item.provenance.timestamp,
            similarity=sim, score=composite, entry_title=meta.title if meta else entry_id,
            context=context,
        ))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def retrieve_concepts(
    question: str, store: Store, embedder: Embedder, *, top_k: int = 3
) -> list[RetrievedConcept]:
    """Rank synthesized OKF concept pages against ``question`` (Phase 18, design report §9
    item 6). Concepts are a *consumer* of the item-level retrieval substrate, not a
    replacement for it (§5 of this repo's canonicalize design) — this only makes concept pages
    independently findable, exactly the way :func:`retrieve` makes raw items findable.

    A concept's score is the better of its stored centroid's similarity to the question and any
    single member item's similarity — blending in members means a concept whose centroid is
    diluted by many loosely-related videos still surfaces when at least one member is a close
    match, the same way one strongly-worded video shouldn't be drowned out by a dozen others.
    """
    qvec = embedder.embed(question)
    vectors_by_key = {
        (entry_id, item_id): vec for item_id, entry_id, vec in store.iter_item_vectors()
    }
    scored: list[RetrievedConcept] = []
    for concept in store.list_concepts():
        centroid = store.concept_centroid(concept.concept_id)
        sim = cosine(qvec, centroid) if centroid else 0.0
        for member in concept.members:
            vec = vectors_by_key.get((member.entry_id, member.item_id))
            if vec is not None:
                sim = max(sim, cosine(qvec, vec))
        scored.append(RetrievedConcept(
            concept_id=concept.concept_id, title=concept.title,
            description=concept.description, similarity=sim, claims=concept.claims,
        ))
    scored.sort(key=lambda c: c.similarity, reverse=True)
    return scored[:top_k]


def retrieve_entities(
    question: str, store: Store, embedder: Embedder, *, top_k: int = 3
) -> list[RetrievedEntity]:
    """Rank synthesized OKF entity pages against ``question`` — the exact ``retrieve_concepts``
    shape, one granularity down (Phase D). Entities are a *consumer* of the item-level retrieval
    substrate, never a replacement for it, same as concepts. No backfill means an older entry
    contributes no entities and simply never appears here — this degrades gracefully, it is not
    an error.
    """
    qvec = embedder.embed(question)
    vectors_by_key = {
        (entry_id, item_id): vec for item_id, entry_id, vec in store.iter_item_vectors()
    }
    scored: list[RetrievedEntity] = []
    for entity in store.list_entities():
        centroid = store.entity_centroid(entity.entity_id)
        sim = cosine(qvec, centroid) if centroid else 0.0
        for member in entity.members:
            vec = vectors_by_key.get((member.entry_id, member.item_id))
            if vec is not None:
                sim = max(sim, cosine(qvec, vec))
        scored.append(RetrievedEntity(
            entity_id=entity.entity_id, kind=entity.kind, title=entity.title,
            description=entity.description, similarity=sim, claims=entity.claims,
        ))
    scored.sort(key=lambda e: e.similarity, reverse=True)
    return scored[:top_k]


def ask(
    question: str,
    store: Store,
    embedder: Embedder,
    client: LLMClient,
    *,
    threshold: float = 0.35,
    top_k: int = 6,
    concept_top_k: int = 3,
    entity_top_k: int = 3,
    lookup_only: bool = False,
    concept_note_depth: str | None = None,
) -> AskResult:
    results = retrieve(question, store, embedder, top_k=top_k)
    concepts = retrieve_concepts(question, store, embedder, top_k=concept_top_k)
    entities = retrieve_entities(question, store, embedder, top_k=entity_top_k)

    # ---- THE GATE: abstain unless an item, a concept, OR an entity clears the SAME threshold,
    # with ZERO synthesis calls either way. Entities never get a lower bar than raw items or
    # concepts (Phase D mirrors Phase 18's rule exactly).
    cleared = [r for r in results if r.similarity >= threshold]
    cleared_concepts = [c for c in concepts if c.similarity >= threshold]
    cleared_entities = [e for e in entities if e.similarity >= threshold]
    if not cleared and not cleared_concepts and not cleared_entities:
        return AskResult(
            abstained=True,
            message="No relevant notes found. Distil answers only from your knowledge base, "
                    "so it won't guess from outside knowledge.",
        )

    # What fed this answer, for the "concepts behind this answer" panel (Phase C) — computed
    # for free from the gate above, no extra retrieval or model call.
    concept_refs = [ConceptRef(c.concept_id, c.title) for c in cleared_concepts]

    # Blend in each cleared concept's/entity's member items — findable even when raw per-item
    # similarity alone would have missed them (design report §9 item 6; Phase D mirrors it).
    all_items = cleared + _recruit_concept_members(
        cleared_concepts, store, exclude={r.item_id for r in cleared}
    )
    all_items += _recruit_entity_members(
        cleared_entities, store, exclude={r.item_id for r in all_items}
    )
    sources = [
        Source(r.item_id, r.entry_id, r.quote, r.timestamp, r.entry_title) for r in all_items
    ]

    # Bare lookup: just the ranked sources, no synthesis call (T-Q5).
    if lookup_only:
        return AskResult(abstained=False, sources=sources, concepts=concept_refs)

    # Grounded synthesis over the cleared items plus any synthesized concept/entity prose — the
    # prose is already grounded (every claim's item_ids resolve to a real member, T-SYN2), so
    # citing it is exactly as trustworthy as citing a directly-retrieved item.
    notes_block = _render_notes(all_items)
    concepts_block = _render_concept_notes(
        cleared_concepts, store, depth=_resolve_concept_note_depth(concept_note_depth)
    )
    if concepts_block:
        notes_block = f"{notes_block}\n\n{concepts_block}" if notes_block else concepts_block
    entities_block = _render_entity_notes(cleared_entities)
    if entities_block:
        notes_block = f"{notes_block}\n\n{entities_block}" if notes_block else entities_block
    raw = client.complete(build_synthesis_prompt(question, notes_block), system=SYSTEM)
    answer, cited, conflict = _parse_synthesis(raw)

    retrieved_ids = {r.item_id for r in all_items}
    grounded = [c for c in cited if c in retrieved_ids]
    ungrounded = [c for c in cited if c not in retrieved_ids]

    # Surface a conflict even if the model didn't, when retrieved items are linked by a
    # `contradicts` edge (T-Q6).
    if not conflict:
        conflict = _detect_contradiction(store, all_items)

    return AskResult(
        abstained=False,
        answer=answer,
        sources=sources,
        cited_item_ids=grounded,
        ungrounded_citations=ungrounded,
        conflict=conflict,
        concepts=concept_refs,
    )


@dataclass
class StreamEvent:
    """One event in a streaming ask. Exactly one field is set per event."""

    kind: str  # "delta" | "abstain" | "final" | "error"
    text: str = ""  # for delta / abstain / error
    result: AskResult | None = None  # for final / abstain


def stream_ask(
    question: str,
    store: Store,
    embedder: Embedder,
    client: LLMClient,
    *,
    threshold: float = 0.35,
    top_k: int = 6,
    concept_top_k: int = 3,
    entity_top_k: int = 3,
    concept_note_depth: str | None = None,
):
    """Streaming sibling of :func:`ask` (WEB_UI_SPEC §9).

    Yields :class:`StreamEvent`:
      * ``abstain`` — nothing cleared the threshold; zero synthesis calls (same gate as ``ask``).
      * ``delta`` — a chunk of answer text, as the model produces it.
      * ``final`` — terminal event carrying the resolved :class:`AskResult` (sources, grounded
        citations, conflict). Sources resolve only after the stream completes.
      * ``error`` — the stream failed partway; callers discard any partial answer and offer retry.

    The synthesis contract is JSON, so raw model chunks are buffered and parsed before any
    answer text is emitted. This prevents the web UI from briefly showing JSON keys or internal
    ``k_01`` citation IDs; those IDs remain available only in the final structured result.
    """
    results = retrieve(question, store, embedder, top_k=top_k)
    concepts = retrieve_concepts(question, store, embedder, top_k=concept_top_k)
    entities = retrieve_entities(question, store, embedder, top_k=entity_top_k)
    cleared = [r for r in results if r.similarity >= threshold]
    cleared_concepts = [c for c in concepts if c.similarity >= threshold]
    cleared_entities = [e for e in entities if e.similarity >= threshold]
    if not cleared and not cleared_concepts and not cleared_entities:
        abstain = AskResult(
            abstained=True,
            message="No relevant notes found. Distil answers only from your knowledge base, "
                    "so it won't guess from outside knowledge.",
        )
        yield StreamEvent(kind="abstain", text=abstain.message, result=abstain)
        return

    concept_refs = [ConceptRef(c.concept_id, c.title) for c in cleared_concepts]

    all_items = cleared + _recruit_concept_members(
        cleared_concepts, store, exclude={r.item_id for r in cleared}
    )
    all_items += _recruit_entity_members(
        cleared_entities, store, exclude={r.item_id for r in all_items}
    )
    sources = [
        Source(r.item_id, r.entry_id, r.quote, r.timestamp, r.entry_title) for r in all_items
    ]
    notes_block = _render_notes(all_items)
    concepts_block = _render_concept_notes(
        cleared_concepts, store, depth=_resolve_concept_note_depth(concept_note_depth)
    )
    if concepts_block:
        notes_block = f"{notes_block}\n\n{concepts_block}" if notes_block else concepts_block
    entities_block = _render_entity_notes(cleared_entities)
    if entities_block:
        notes_block = f"{notes_block}\n\n{entities_block}" if notes_block else entities_block
    prompt = build_synthesis_prompt(question, notes_block)

    chunks: list[str] = []
    try:
        stream_iter = client.stream(prompt, system=SYSTEM)
    except (AttributeError, NotImplementedError):
        # Fallback: client only implements complete() — yield the full response as one chunk
        stream_iter = iter([client.complete(prompt, system=SYSTEM)])

    try:
        for delta in stream_iter:
            if not delta:
                continue
            chunks.append(delta)
    except Exception as exc:  # WEB_UI_SPEC §9: discard partial, signal retry
        yield StreamEvent(kind="error", text=str(exc) or exc.__class__.__name__)
        return

    raw = "".join(chunks)
    answer, cited, conflict = _parse_synthesis(raw)
    if answer:
        yield StreamEvent(kind="delta", text=answer)
    retrieved_ids = {r.item_id for r in all_items}
    grounded = [c for c in cited if c in retrieved_ids]
    ungrounded = [c for c in cited if c not in retrieved_ids]
    if not conflict:
        conflict = _detect_contradiction(store, all_items)

    yield StreamEvent(
        kind="final",
        result=AskResult(
            abstained=False, answer=answer, sources=sources,
            cited_item_ids=grounded, ungrounded_citations=ungrounded, conflict=conflict,
            concepts=concept_refs,
        ),
    )


# ---- helpers ----------------------------------------------------------------------------


def _feedback_multiplier(score: int | None) -> float:
    # Unscored → neutral 1.0; 5★ → 1.4, 1★ → 0.6 (monotonic, bounded).
    if score is None:
        return 1.0
    return 0.6 + (score - 1) * (0.8 / 4)


def _recency_multiplier(created_at: str | None, now: datetime) -> float:
    if not created_at:
        return 1.0
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return 1.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 + 0.5 * math.exp(-age_days / _RECENCY_HALF_LIFE_DAYS)


def _load_item(store: Store, entry_id: str, item_id: str):
    try:
        entry = store.load_entry(entry_id)
    except (FileNotFoundError, ValueError):
        return None
    for item in entry.knowledge_items:
        if item.item_id == item_id:
            return item, Store.note_context_for_item(entry, item_id)
    return None


def _recruit_concept_members(
    concepts: list[RetrievedConcept], store: Store, *, exclude: set[str]
) -> list[RetrievedItem]:
    """Pull each cleared concept's member items into the evidence pool (Phase 18). Every
    recruited item still resolves to a real :class:`KnowledgeItem` via :func:`_load_item`,
    exactly like a directly-retrieved item — concepts only widen *which* items are considered,
    never how an item earns a place in ``sources``/``retrieved_ids``.
    """
    if not concepts:
        return []
    entry_meta = {r.entry_id: r for r in store.list_entries()}
    seen = set(exclude)
    out: list[RetrievedItem] = []
    for concept in concepts:
        full = store.load_concept(concept.concept_id)
        if full is None:
            continue
        for member in full.members:
            if member.item_id in seen:
                continue
            seen.add(member.item_id)
            loaded = _load_item(store, member.entry_id, member.item_id)
            if loaded is None:
                continue
            item, context = loaded
            meta = entry_meta.get(member.entry_id)
            out.append(RetrievedItem(
                item_id=member.item_id, entry_id=member.entry_id, statement=item.statement,
                quote=item.provenance.quote, timestamp=item.provenance.timestamp,
                similarity=concept.similarity, score=concept.similarity,
                entry_title=meta.title if meta else member.entry_id, context=context,
            ))
    return out


def _resolve_concept_note_depth(override: str | None) -> str:
    """``concept_note_depth`` kwarg wins; else ``DISTIL_CONCEPT_NOTE_DEPTH`` env, read at call
    time (same DI-friendly pattern as the other retrieval knobs in ``store.py``); an unrecognized
    value falls back to :data:`CONCEPT_NOTE_DEPTH_DEFAULT` rather than raising."""
    depth = (override or os.environ.get("DISTIL_CONCEPT_NOTE_DEPTH", CONCEPT_NOTE_DEPTH_DEFAULT))
    depth = depth.strip().lower()
    return depth if depth in _CONCEPT_NOTE_DEPTHS else CONCEPT_NOTE_DEPTH_DEFAULT


def _recruit_entity_members(
    entities: list[RetrievedEntity], store: Store, *, exclude: set[str]
) -> list[RetrievedItem]:
    """Pull each cleared entity's member items into the evidence pool — the exact
    ``_recruit_concept_members`` shape, one granularity down (Phase D)."""
    if not entities:
        return []
    entry_meta = {r.entry_id: r for r in store.list_entries()}
    seen = set(exclude)
    out: list[RetrievedItem] = []
    for entity in entities:
        full = store.load_entity(entity.entity_id)
        if full is None:
            continue
        for member in full.members:
            if member.item_id in seen:
                continue
            seen.add(member.item_id)
            loaded = _load_item(store, member.entry_id, member.item_id)
            if loaded is None:
                continue
            item, context = loaded
            meta = entry_meta.get(member.entry_id)
            out.append(RetrievedItem(
                item_id=member.item_id, entry_id=member.entry_id, statement=item.statement,
                quote=item.provenance.quote, timestamp=item.provenance.timestamp,
                similarity=entity.similarity, score=entity.similarity,
                entry_title=meta.title if meta else member.entry_id, context=context,
            ))
    return out


def _render_entity_notes(entities: list[RetrievedEntity]) -> str:
    """Pre-synthesized entity-page prose as extra evidence — the exact ``_render_concept_notes``
    shape, one granularity down (Phase D)."""
    lines: list[str] = []
    for e in entities:
        if not e.claims:
            continue
        lines.append(f'Entity "{e.title}" ({e.kind}): {e.description}')
        for claim in e.claims:
            markers = "".join(f"[{item_id}]" for item_id in claim.item_ids)
            lines.append(f"  {claim.text} {markers}".rstrip())
    return "\n".join(lines)


def _render_concept_notes(
    concepts: list[RetrievedConcept], store: Store, *, depth: str = CONCEPT_NOTE_DEPTH_DEFAULT
) -> str:
    """Pre-synthesized concept-page prose as extra evidence (Phase 18; depth added Phase C).
    Already grounded — every ``ConceptClaim.item_ids`` resolves to a real member (T-SYN2) — so
    citing one of these claims is exactly as trustworthy as citing a raw retrieved item; nothing
    here is model text trusted as a citation, it's the same ``[item_id]`` marker convention
    :func:`_render_notes` already uses.

    ``depth="full"`` additionally inlines each cited member's own quote beneath its claim — the
    same ``ConceptMember.quote`` copied verbatim from real provenance at match time that the
    concept detail page and the OKF page both cite, i.e. the substance a reader would see after
    following the link to the concept page, not just its claim summary. This never reaches
    beyond ``concept.members``/``concept.claims`` (no edge traversal, no member outside this
    concept's own membership), so grounding is unaffected either way.
    """
    lines: list[str] = []
    for c in concepts:
        if not c.claims:
            continue
        lines.append(f'Concept "{c.title}": {c.description}')
        members_by_item = {}
        if depth == "full":
            full = store.load_concept(c.concept_id)
            if full is not None:
                members_by_item = {m.item_id: m for m in full.members}
        for claim in c.claims:
            markers = "".join(f"[{item_id}]" for item_id in claim.item_ids)
            lines.append(f"  {claim.text} {markers}".rstrip())
            if depth == "full":
                for item_id in claim.item_ids:
                    member = members_by_item.get(item_id)
                    if member is None:
                        continue
                    ts = f" @ {member.timestamp}" if member.timestamp else ""
                    lines.append(f'    [{item_id}] quote: "{member.quote}"{ts}')
    return "\n".join(lines)


def _render_notes(items: list[RetrievedItem]) -> str:
    lines = []
    for r in items:
        line = f"[{r.item_id}] {r.statement} (quote: \"{r.quote}\")"
        if r.context:
            line += f"\n  synthesized note context: {r.context}"
        lines.append(line)
    return "\n".join(lines)


def _parse_synthesis(raw: str) -> tuple[str, list[str], str | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip()
        cited = list(data.get("cited_item_ids", []))
        conflict = data.get("conflict") or None
        # Backfill citations from inline [id] markers if the field is empty.
        if not cited:
            cited = _CITATION.findall(answer)
        return _clean_answer_text(answer), cited, conflict
    except json.JSONDecodeError:
        # Degrade gracefully: treat the whole response as the answer; extract inline cites.
        return _clean_answer_text(text), _CITATION.findall(text), None


def _clean_answer_text(answer: str) -> str:
    """Remove internal item-id citation markers from reader-facing answer text."""
    text = _CITATION.sub("", answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_contradiction(store: Store, items: list[RetrievedItem]) -> str | None:
    entry_ids = {r.entry_id for r in items}
    for entry_id in entry_ids:
        try:
            entry = store.load_entry(entry_id)
        except (FileNotFoundError, ValueError):
            continue
        for edge in entry.related_entries:
            if edge.relation == "contradicts" and edge.target in entry_ids:
                return (
                    f"Your notes disagree: {entry_id} contradicts {edge.target}. "
                    "Both are shown above."
                )
    return None
