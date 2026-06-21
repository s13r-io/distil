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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .embed import Embedder
from .llm import LLMClient
from .prompts.synthesize import SYSTEM, build_synthesis_prompt
from .store import Store

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CITATION = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# Ranking weights for similarity × feedback × recency.
_RECENCY_HALF_LIFE_DAYS = 180.0


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
class Source:
    item_id: str
    entry_id: str
    quote: str
    timestamp: str | None
    entry_title: str = ""


@dataclass
class AskResult:
    abstained: bool
    message: str = ""
    answer: str | None = None
    sources: list[Source] = field(default_factory=list)
    cited_item_ids: list[str] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    conflict: str | None = None


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


def ask(
    question: str,
    store: Store,
    embedder: Embedder,
    client: LLMClient,
    *,
    threshold: float = 0.35,
    top_k: int = 6,
    lookup_only: bool = False,
) -> AskResult:
    results = retrieve(question, store, embedder, top_k=top_k)

    # ---- THE GATE: abstain if nothing clears the threshold, with ZERO synthesis calls. ----
    cleared = [r for r in results if r.similarity >= threshold]
    if not cleared:
        return AskResult(
            abstained=True,
            message="No relevant notes found. Distil answers only from your knowledge base, "
                    "so it won't guess from outside knowledge.",
        )

    sources = [Source(r.item_id, r.entry_id, r.quote, r.timestamp, r.entry_title) for r in cleared]

    # Bare lookup: just the ranked sources, no synthesis call (T-Q5).
    if lookup_only:
        return AskResult(abstained=False, sources=sources)

    # Grounded synthesis over the cleared items only.
    notes_block = _render_notes(cleared)
    raw = client.complete(build_synthesis_prompt(question, notes_block), system=SYSTEM)
    answer, cited, conflict = _parse_synthesis(raw)

    retrieved_ids = {r.item_id for r in cleared}
    grounded = [c for c in cited if c in retrieved_ids]
    ungrounded = [c for c in cited if c not in retrieved_ids]

    # Surface a conflict even if the model didn't, when retrieved items are linked by a
    # `contradicts` edge (T-Q6).
    if not conflict:
        conflict = _detect_contradiction(store, cleared)

    return AskResult(
        abstained=False,
        answer=answer,
        sources=sources,
        cited_item_ids=grounded,
        ungrounded_citations=ungrounded,
        conflict=conflict,
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
    cleared = [r for r in results if r.similarity >= threshold]
    if not cleared:
        abstain = AskResult(
            abstained=True,
            message="No relevant notes found. Distil answers only from your knowledge base, "
                    "so it won't guess from outside knowledge.",
        )
        yield StreamEvent(kind="abstain", text=abstain.message, result=abstain)
        return

    sources = [Source(r.item_id, r.entry_id, r.quote, r.timestamp, r.entry_title) for r in cleared]
    notes_block = _render_notes(cleared)
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
    retrieved_ids = {r.item_id for r in cleared}
    grounded = [c for c in cited if c in retrieved_ids]
    ungrounded = [c for c in cited if c not in retrieved_ids]
    if not conflict:
        conflict = _detect_contradiction(store, cleared)

    yield StreamEvent(
        kind="final",
        result=AskResult(
            abstained=False, answer=answer, sources=sources,
            cited_item_ids=grounded, ungrounded_citations=ungrounded, conflict=conflict,
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
