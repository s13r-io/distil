"""Per-video OKF export layer (Phase 2). ARCHITECTURE.md; SCHEMA.md (Distil's, not OKF's).

Distil's ``kb/<entry_id>.md`` (JSON front matter, lossless) stays the internal source of
truth. This module derives a second, neutral layer next to it — an `Open Knowledge Format
<https://github.com/GoogleCloudPlatform/knowledge-catalog>`_ (OKF v0.1) bundle — so a filed
:class:`~distil.models.KBEntry` can be opened in OpenKnowledge or shared without leaking
personal state (feedback scores, application links). Every export is a full, deterministic
regeneration: re-exporting the same entry rewrites the same files byte-for-byte (given the
same inputs and day), so filing is idempotent and safe to repeat.

Bundle layout, rooted at ``okf_root`` (default: a sibling of ``kb_dir`` named ``okf`` — the
same "sibling of kb/" convention Distil already uses for ``data/``, so the two on-disk stores
stay easy to find together without one nesting inside the other)::

    okf_root/
      index.md            # bundle root catalog; declares okf_version
      sources/
        index.md          # one line per source, from its `description` frontmatter
        <slug>.md          # neutral summary + key moments, links to raw/
      raw/
        <slug>.md          # immutable timestamped transcript

This module now also exports ``concepts/`` (Phase 15, cross-video idea synthesis) and
``entities/`` (Phase D, cross-video tool/person/organization synthesis) alongside the ``sources/``
+ ``raw/`` layer above — see ``export_concept``/``export_entity``. OpenKnowledge wiring proper
remains out of scope.

Slug derivation (stable identity — SCHEMA.md §2 "the path is the identity"): the video's
``source.title`` is slugified (lowercased, non-alnum runs collapsed to single hyphens,
stripped). If that yields nothing usable (empty/untitled source), the slug falls back to the
entry's ``entry_id``. The title is not expected to change once an entry is filed, so the slug
stays stable across re-exports of the same entry.

Collision handling: each ``sources/<slug>.md`` records its owning entry in a
``distil_entry_id`` frontmatter field. When resolving a slug for export/removal (i.e. when an
``okf_root`` is supplied), that field is consulted first — if this entry already owns a slug
from a previous export, that slug is reused even if another entry has since taken the base
slug. Otherwise, if the base (title-derived) slug is free or already owned by this same entry,
it is used as-is. Only when the base slug is owned by a *different* entry_id does the slug
gain a short suffix derived from this entry's entry_id, so two distinct entries with the same
title never overwrite each other's pages.

``published`` is not fetched by Distil today (YouTube oEmbed does not return a publish date),
so it is set to the entry's capture date as the closest available honest proxy; this is
documented here rather than invented silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .ingest import Transcript
from .models import Concept, ConceptEdge, Entity, KBEntry
from .source import _youtube_video_id, display_title
from .synthesize_concept import find_claim_contradictions, render_claim

_EDGE_HEADINGS: tuple[tuple[str, str], ...] = (
    ("contrasts_with", "Contrasts with"),
    ("builds_on", "Builds on"),
    ("related", "Related"),
)

if TYPE_CHECKING:  # pragma: no cover - avoids an okf<->store import cycle (store imports okf)
    from .store import Store

_SLUG_RUN = re.compile(r"[^a-z0-9]+")
_MAX_CONCEPT_TAGS = 8


def slug_for_entry(entry: KBEntry, okf_root: str | Path | None = None) -> str:
    """Deterministic, stable OKF slug for ``entry`` (see module docstring for the collision rule).

    Without ``okf_root`` this just computes the base (title-derived, or entry_id-fallback)
    slug with no collision check — used by callers that only need the "natural" slug shape.
    With ``okf_root``, existing ``sources/*.md`` frontmatter is consulted to keep the slug
    stable across re-exports and to disambiguate title collisions between distinct entries.
    """
    base = slugify(entry.source.title) or entry.entry_id
    if okf_root is None:
        return base

    sources_dir = Path(okf_root) / "sources"
    owned = _slug_owned_by(sources_dir, entry.entry_id)
    if owned:
        return owned

    owner = _owner_of_slug(sources_dir, base)
    if owner is None or owner == entry.entry_id:
        return base

    suffix_len = 6
    while True:
        suffix = slugify(entry.entry_id[-suffix_len:]) or entry.entry_id
        candidate = f"{base}-{suffix}"
        owner = _owner_of_slug(sources_dir, candidate)
        if owner is None or owner == entry.entry_id:
            return candidate
        suffix_len += 2


def _slug_owned_by(sources_dir: Path, entry_id: str) -> str | None:
    if not sources_dir.exists():
        return None
    for path in sources_dir.glob("*.md"):
        if path.name == "index.md":
            continue
        if _frontmatter_field(path.read_text(encoding="utf-8"), "distil_entry_id") == entry_id:
            return path.stem
    return None


def _owner_of_slug(sources_dir: Path, slug: str) -> str | None:
    path = sources_dir / f"{slug}.md"
    if not path.exists():
        return None
    return _frontmatter_field(path.read_text(encoding="utf-8"), "distil_entry_id")


def slugify(text: str) -> str:
    """Lowercase, hyphenate, and strip ``text`` into a URL-safe slug fragment.

    Shared by :func:`slug_for_entry` and, for concept titles, ``canonicalize.py`` — the one
    slugification rule every OKF path-identity slug in Distil derives from.
    """
    return _SLUG_RUN.sub("-", text.strip().lower()).strip("-")


def export_entry(entry: KBEntry, transcript: Transcript, okf_root: str | Path) -> None:
    """Write/refresh ``sources/<slug>.md`` + ``raw/<slug>.md`` and regenerate both indexes.

    Stage 7 (File) runs before canonicalize (Stage 8, design report §5), so this page is
    always rendered with no "## Concepts covered" section — :func:`render_source_with_concepts`
    is the post-canonicalize step that adds it once concept membership is known.
    """
    root = Path(okf_root)
    sources_dir = root / "sources"
    raw_dir = root / "raw"
    sources_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    slug = slug_for_entry(entry, root)
    (raw_dir / f"{slug}.md").write_text(_render_raw(entry, transcript, slug), encoding="utf-8")
    (sources_dir / f"{slug}.md").write_text(_render_source(entry, slug), encoding="utf-8")
    _rebuild_indexes(root)


def remove_entry(entry: KBEntry, okf_root: str | Path) -> None:
    """Delete an entry's OKF pages (if present) and regenerate both indexes."""
    remove_entry_pages(slug_for_entry(entry, okf_root), okf_root)


def remove_entry_pages(slug: str, okf_root: str | Path) -> None:
    """Delete ``sources/<slug>.md`` and ``raw/<slug>.md`` (if present) and regenerate both
    indexes. The slug-only counterpart to :func:`remove_entry`, for callers (delete-cascade
    orchestration) that need to remove an entry's pages without a loadable ``KBEntry`` — see
    :func:`find_slug_for_entry_id` for recovering the slug in that case.
    """
    root = Path(okf_root)
    (root / "sources" / f"{slug}.md").unlink(missing_ok=True)
    (root / "raw" / f"{slug}.md").unlink(missing_ok=True)
    _rebuild_indexes(root)


def find_slug_for_entry_id(entry_id: str, okf_root: str | Path) -> str | None:
    """Recover a ``sources/<slug>.md`` page's slug from its ``distil_entry_id`` frontmatter
    alone — no loadable :class:`KBEntry` needed. This is the same lookup
    :func:`slug_for_entry` uses internally to keep a slug stable across re-exports; it is
    exposed here for callers (e.g. delete-cascade orchestration) that only have an ``entry_id``
    because the ``kb/<id>.md`` file is missing or failed to parse.
    """
    return _slug_owned_by(Path(okf_root) / "sources", entry_id)


def rebuild_indexes(okf_root: str | Path) -> None:
    """Public entry point for regenerating ``index.md``/``sources/index.md``/``concepts/
    index.md`` from whatever pages currently exist on disk — used by reconcile after removing
    orphaned files so the indexes reflect the repaired bundle."""
    _rebuild_indexes(Path(okf_root))


def frontmatter_field(text: str, key: str) -> str | None:
    """Public wrapper over the frontmatter scan every page type in this module shares —
    exposed for callers outside this module (reconcile) that need to read a page's own
    frontmatter without duplicating the parsing rule."""
    return _frontmatter_field(text, key)


def render_source_with_concepts(entry: KBEntry, store: Store, okf_root: str | Path) -> None:
    """Re-render ``sources/<slug>.md`` for ``entry``, adding a "## Concepts covered" section
    listing every concept with a member from this entry (SCHEMA §5 "bidirectional"; design
    report §3, §5), and — Phase D — a "## Entities mentioned" section the same way. A
    post-canonicalize step: Stage 7 (File) already wrote this page without concept/entity
    knowledge, so this re-touches it once membership is known. Full deterministic regeneration
    like the rest of this module — re-rendering the same concept/entity set is byte-identical.
    """
    root = Path(okf_root)
    slug = slug_for_entry(entry, root)
    covering = sorted(
        (c for c in store.list_concepts() if any(m.entry_id == entry.entry_id for m in c.members)),
        key=lambda c: c.title.lower(),
    )
    covering_entities = sorted(
        (e for e in store.list_entities() if any(m.entry_id == entry.entry_id for m in e.members)),
        key=lambda e: e.title.lower(),
    )
    text = _render_source(
        entry, slug, covering_concepts=covering, covering_entities=covering_entities
    )
    (root / "sources" / f"{slug}.md").write_text(text, encoding="utf-8")


def export_concept(concept: Concept, store: Store, okf_root: str | Path) -> None:
    """Write/refresh ``concepts/<concept_id>.md`` and regenerate all indexes (design report §3).

    No feedback/application-link data by construction (``Concept`` carries none). Claims are
    rendered with code-assembled citations (:func:`distil.synthesize_concept.render_claim`) —
    never model-authored citation text.
    """
    root = Path(okf_root)
    concepts_dir = root / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / f"{concept.concept_id}.md").write_text(
        _render_concept(concept, store, root), encoding="utf-8"
    )
    _rebuild_indexes(root)


def remove_concept(concept_id: str, okf_root: str | Path) -> None:
    """Delete a concept's OKF page (if present) and regenerate all indexes.

    Called whenever a concept drops to zero members after a retraction (design report §5) —
    the DB row is already gone (``Store.retract_entry_concept_memberships``); this cleans up
    the page it left behind.
    """
    root = Path(okf_root)
    (root / "concepts" / f"{concept_id}.md").unlink(missing_ok=True)
    _rebuild_indexes(root)


def export_entity(entity: Entity, store: Store, okf_root: str | Path) -> None:
    """Write/refresh ``entities/<entity_id>.md`` and regenerate all indexes — the exact
    ``export_concept`` shape, one granularity down (Phase D)."""
    root = Path(okf_root)
    entities_dir = root / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    (entities_dir / f"{entity.entity_id}.md").write_text(
        _render_entity(entity, store, root), encoding="utf-8"
    )
    _rebuild_indexes(root)


def remove_entity(entity_id: str, okf_root: str | Path) -> None:
    """Delete an entity's OKF page (if present) and regenerate all indexes — the exact
    ``remove_concept`` shape, called whenever an entity drops to zero members."""
    root = Path(okf_root)
    (root / "entities" / f"{entity_id}.md").unlink(missing_ok=True)
    _rebuild_indexes(root)


# ---- page rendering ----------------------------------------------------------------------


def _render_source(
    entry: KBEntry,
    slug: str,
    covering_concepts: list[Concept] | None = None,
    covering_entities: list[Entity] | None = None,
) -> str:
    title = display_title(
        entry.source.title,
        entry.distilled_note.title if entry.distilled_note is not None else None,
    )
    thesis = (
        entry.distilled_note.core_takeaway.text
        if entry.distilled_note is not None
        else entry.source.title
    )
    youtube_id = _youtube_id(entry.source.url)

    lines = ["---", "type: source", f"title: {_yaml_str(title)}", f"description: {_yaml_str(thesis)}"]
    if youtube_id:
        lines.append(f"youtube_id: {youtube_id}")
    if entry.source.url:
        lines.append(f"url: {entry.source.url}")
    lines.append(f"slug: {slug}")
    lines.append(f"distil_entry_id: {entry.entry_id}")
    lines.append(f"published: {_date_only(entry.source.captured_at)}")
    lines.append(f"duration: {_yaml_str(_format_duration(entry.source.duration_sec))}")
    lines.append(f"raw: {_yaml_str(f'../raw/{slug}.md')}")
    lines.append(f"tags: {_yaml_flow_list(entry.tags.topics)}")
    lines.append(f"created: {_date_only(entry.meta.created_at)}")
    lines.append(f"updated: {_date_only(entry.meta.created_at)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(thesis)
    lines.append("")

    moments = sorted(
        (item for item in entry.knowledge_items if item.provenance.timestamp),
        key=lambda item: item.provenance.timestamp,
    )
    if moments:
        lines.append("## Key moments")
        lines.append("")
        for item in moments:
            lines.append(f"- **[{item.provenance.timestamp}]** {item.statement}")
        lines.append("")

    lines.append("## Transcript")
    lines.append("")
    lines.append(f"[Raw transcript](../raw/{slug}.md)")
    lines.append("")

    if covering_concepts:
        lines.append("## Concepts covered")
        lines.append("")
        for concept in covering_concepts:
            lines.append(f"- [{concept.title}](../concepts/{concept.concept_id}.md)")
        lines.append("")

    if covering_entities:
        lines.append("## Entities mentioned")
        lines.append("")
        for entity in covering_entities:
            lines.append(f"- [{entity.title}](../entities/{entity.entity_id}.md)")
        lines.append("")
    return "\n".join(lines)


def _render_concept(concept: Concept, store: Store, root: Path) -> str:
    member_entries: dict[str, KBEntry] = {}
    for member in concept.members:
        if member.entry_id in member_entries:
            continue
        try:
            member_entries[member.entry_id] = store.load_entry(member.entry_id)
        except Exception:
            continue

    tags = _aggregate_concept_tags(member_entries.values())
    videos = sorted({slug_for_entry(entry, root) for entry in member_entries.values()})

    lines = [
        "---",
        "type: concept",
        f"title: {_yaml_str(concept.title)}",
        f"description: {_yaml_str(concept.description)}",
        f"tags: {_yaml_flow_list(tags)}",
        f"videos: {_yaml_flow_list(videos)}",
        f"created: {_date_only(concept.created_at)}",
        f"updated: {_date_only(concept.updated_at)}",
        "---",
        "",
        f"# {concept.title}",
        "",
    ]

    citations = _concept_citations(concept, member_entries, root)
    if concept.claims:
        contradictions = find_claim_contradictions(concept, store)
        lines.append("## Claims")
        lines.append("")
        for idx, claim in enumerate(concept.claims):
            lines.append(render_claim(claim, citations))
            rows = contradictions.get(idx)
            if rows:
                lines.append(_render_contradiction(rows, member_entries, root))
            lines.append("")
    else:
        lines.append(concept.description)
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen_entries: set[str] = set()
    for member in concept.members:
        if member.entry_id in seen_entries:
            continue
        seen_entries.add(member.entry_id)
        entry = member_entries.get(member.entry_id)
        if entry is None:
            continue
        slug = slug_for_entry(entry, root)
        quote = f"[{member.timestamp}] {member.quote}" if member.timestamp else member.quote
        lines.append(f'- [{entry.source.title}](../sources/{slug}.md) - "{quote}"')
    lines.append("")

    if concept.edges:
        other_titles = {
            c.concept_id: c.title for c in store.list_concepts() if c.concept_id != concept.concept_id
        }
        for relation, heading in _EDGE_HEADINGS:
            _append_edge_section(lines, heading, concept.edges, relation, other_titles)
    return "\n".join(lines)


def _render_contradiction(
    rows: list[tuple[str, str, str]], member_entries: dict[str, KBEntry], root: Path
) -> str:
    """Deterministic ``> **Contradiction:**`` flag (Phase 16 task brief): names every member
    whose ``stance`` disagreed with another member cited by the same claim, resolved to its OKF
    slug from verified member/entry data — never model text, same discipline as citations."""
    parts: list[str] = []
    for entry_id, _item_id, stance in rows:
        entry = member_entries.get(entry_id)
        label = slug_for_entry(entry, root) if entry is not None else entry_id
        parts.append(f"{label} ({stance})")
    return f"> **Contradiction:** members disagree — {', '.join(parts)}."


def _append_edge_section(
    lines: list[str],
    heading: str,
    edges: list[ConceptEdge],
    relation: str,
    titles: dict[str, str],
) -> None:
    """Render one typed-edge section (``## Contrasts with`` / ``## Builds on`` / ``## Related``,
    Phase 16 design report §9 item 4) if ``edges`` has at least one link of ``relation`` whose
    target concept still exists. Links are same-directory (``concepts/`` -> ``concepts/``)."""
    matching = [e for e in edges if e.relation == relation and e.target_concept_id in titles]
    if not matching:
        return
    lines.append(f"## {heading}")
    lines.append("")
    for edge in matching:
        lines.append(f"- [{titles[edge.target_concept_id]}]({edge.target_concept_id}.md)")
    lines.append("")


def _render_entity(entity: Entity, store: Store, root: Path) -> str:
    """Render an entity's OKF page — the exact ``_render_concept`` shape, one granularity down,
    minus the typed-edge section (Phase D has no entity<->entity edges) and plus a ``kind``
    frontmatter field."""
    member_entries: dict[str, KBEntry] = {}
    for member in entity.members:
        if member.entry_id in member_entries:
            continue
        try:
            member_entries[member.entry_id] = store.load_entry(member.entry_id)
        except Exception:
            continue

    videos = sorted({slug_for_entry(entry, root) for entry in member_entries.values()})

    lines = [
        "---",
        "type: entity",
        f"kind: {entity.kind}",
        f"title: {_yaml_str(entity.title)}",
        f"description: {_yaml_str(entity.description)}",
        f"videos: {_yaml_flow_list(videos)}",
        f"created: {_date_only(entity.created_at)}",
        f"updated: {_date_only(entity.updated_at)}",
        "---",
        "",
        f"# {entity.title}",
        "",
    ]

    citations = _concept_citations(entity, member_entries, root)
    if entity.claims:
        lines.append("## Claims")
        lines.append("")
        for claim in entity.claims:
            lines.append(render_claim(claim, citations))
            lines.append("")
    else:
        lines.append(entity.description)
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen_entries: set[str] = set()
    for member in entity.members:
        if member.entry_id in seen_entries:
            continue
        seen_entries.add(member.entry_id)
        entry = member_entries.get(member.entry_id)
        if entry is None:
            continue
        slug = slug_for_entry(entry, root)
        quote = f"[{member.timestamp}] {member.quote}" if member.timestamp else member.quote
        lines.append(f'- [{entry.source.title}](../sources/{slug}.md) - "{quote}"')
    lines.append("")
    return "\n".join(lines)


def _aggregate_concept_tags(entries) -> list[str]:
    """Union of member entries' ``tags.topics``, deduped, first-seen order, capped at
    ``_MAX_CONCEPT_TAGS`` (open question, design report: unbounded growth as a concept
    accumulates videos with varied tags — capped the same way ``note._normalize_topics`` is)."""
    tags: list[str] = []
    for entry in entries:
        for topic in entry.tags.topics:
            if topic not in tags:
                tags.append(topic)
    return tags[:_MAX_CONCEPT_TAGS]


def _concept_citations(
    concept: Concept | Entity, member_entries: dict[str, KBEntry], root: Path
) -> dict[str, tuple[str, str | None]]:
    """``item_id`` -> ``(okf_slug, timestamp)``, resolved from verified member/entry data —
    never from model text (design report §4 step 4). Works identically for a ``Concept`` or an
    ``Entity`` — both share the same ``members[i].entry_id/item_id/timestamp`` shape."""
    citations: dict[str, tuple[str, str | None]] = {}
    for member in concept.members:
        entry = member_entries.get(member.entry_id)
        if entry is None:
            continue
        citations[member.item_id] = (slug_for_entry(entry, root), member.timestamp)
    return citations


def _render_raw(entry: KBEntry, transcript: Transcript, slug: str) -> str:
    youtube_id = _youtube_id(entry.source.url)

    lines = ["---", "type: raw-transcript", f"title: {_yaml_str(entry.source.title)}"]
    if youtube_id:
        lines.append(f"youtube_id: {youtube_id}")
    if entry.source.url:
        lines.append(f"url: {entry.source.url}")
    lines.append(f"slug: {slug}")
    lines.append(f"fetched_at: {_date_only(entry.meta.created_at)}")
    lines.append("immutable: true")
    lines.append("---")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    for segment in transcript.segments:
        label = segment.timestamp or segment.locator
        lines.append(f"**[{label}]** {segment.text}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


_RAW_SEGMENT_LINE = re.compile(r"^\*\*\[.*?\]\*\*\s?(.*)$")


def load_raw_transcript_text(slug: str, okf_root: str | Path) -> str | None:
    """Recover the transcript's plain text back out of ``raw/<slug>.md`` (the immutable
    per-video page :func:`export_entry` writes) — the only persisted copy of a filed entry's
    transcript. Used by the narrative-summary refresh action, which must work from stored
    text and never re-fetch. Returns ``None`` when the page doesn't exist (an entry filed
    before OKF export existed, or one whose raw page was later removed by ``reconcile.py``) —
    a real, expected state this system produces, not a bug to work around."""
    path = Path(okf_root) / "raw" / f"{slug}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    body = text.split("## Transcript", 1)
    if len(body) < 2:
        return None
    lines: list[str] = []
    for line in body[1].splitlines():
        match = _RAW_SEGMENT_LINE.match(line.strip())
        if match and match.group(1):
            lines.append(match.group(1))
    if not lines:
        return None
    return " ".join(lines)


# ---- indexes -------------------------------------------------------------------------------


def _rebuild_indexes(root: Path) -> None:
    sources_entries = _collect_pages(root / "sources")
    concepts_entries = sorted(_collect_pages(root / "concepts"), key=lambda e: e[1].lower())
    entities_entries = sorted(_collect_pages(root / "entities"), key=lambda e: e[1].lower())

    sources_index = ["# Sources", ""]
    for slug, title, description in sources_entries:
        sources_index.append(f"- [{title}]({slug}.md) - {description}")
    sources_index.append("")
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "sources" / "index.md").write_text("\n".join(sources_index), encoding="utf-8")

    concepts_index = ["# Concepts", ""]
    for slug, title, description in concepts_entries:
        concepts_index.append(f"- [{title}]({slug}.md) - {description}")
    concepts_index.append("")
    (root / "concepts").mkdir(parents=True, exist_ok=True)
    (root / "concepts" / "index.md").write_text("\n".join(concepts_index), encoding="utf-8")

    entities_index = ["# Entities", ""]
    for slug, title, description in entities_entries:
        entities_index.append(f"- [{title}]({slug}.md) - {description}")
    entities_index.append("")
    (root / "entities").mkdir(parents=True, exist_ok=True)
    (root / "entities" / "index.md").write_text("\n".join(entities_index), encoding="utf-8")

    root_index = [
        "---",
        'okf_version: "0.1"',
        "---",
        "",
        "# Distil OKF Bundle",
        "",
        "Per-video OKF layer generated by Distil, alongside cross-video `concepts/` pages "
        "synthesized from canonicalized knowledge items, and `entities/` pages — tools, people, "
        "and organizations named across videos, canonicalized the same way. Entities are only "
        "extracted for newly ingested videos (no backfill), so an older entry may contribute to "
        "no entity pages; this bundle covers `sources/` (neutral per-video summaries), `raw/` "
        "(immutable transcripts), `concepts/` (cross-video idea synthesis), and `entities/` "
        "(cross-video tool/person/organization synthesis).",
        "",
        "## Sources",
        "",
    ]
    for slug, title, _description in sources_entries:
        root_index.append(f"- [{title}](sources/{slug}.md)")
    root_index.append("")
    root_index.append("See [sources/index.md](sources/index.md) for one-line descriptions.")
    root_index.append("")
    root_index.append("## Concepts")
    root_index.append("")
    for slug, title, _description in concepts_entries:
        root_index.append(f"- [{title}](concepts/{slug}.md)")
    root_index.append("")
    root_index.append("See [concepts/index.md](concepts/index.md) for one-line descriptions.")
    root_index.append("")
    root_index.append("## Entities")
    root_index.append("")
    for slug, title, _description in entities_entries:
        root_index.append(f"- [{title}](entities/{slug}.md)")
    root_index.append("")
    root_index.append("See [entities/index.md](entities/index.md) for one-line descriptions.")
    root_index.append("")
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.md").write_text("\n".join(root_index), encoding="utf-8")


def _collect_pages(dirpath: Path) -> list[tuple[str, str, str]]:
    """One ``(slug, title, description)`` tuple per non-index page in ``dirpath``, reused for
    both ``sources/`` and ``concepts/`` — same frontmatter-scan shape either way."""
    entries: list[tuple[str, str, str]] = []
    if dirpath.exists():
        for path in sorted(dirpath.glob("*.md")):
            if path.name == "index.md":
                continue
            text = path.read_text(encoding="utf-8")
            title = _frontmatter_field(text, "title") or path.stem
            description = _frontmatter_field(text, "description") or ""
            entries.append((path.stem, title, description))
    return entries


# ---- small helpers ---------------------------------------------------------------------


def _youtube_id(url: str | None) -> str | None:
    if not url:
        return None
    return _youtube_video_id(urlparse(url))


def _format_duration(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _date_only(iso_timestamp: str) -> str:
    return iso_timestamp[:10] if len(iso_timestamp) >= 10 else iso_timestamp


def _yaml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return f'"{escaped}"'


def _yaml_flow_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def _frontmatter_field(text: str, key: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    front = text[3:end] if end != -1 else ""
    prefix = f"{key}:"
    for line in front.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            return value
    return None
