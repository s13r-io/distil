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

Concepts/entities layers (Phase 3/5), the canonicalize engine, and OpenKnowledge wiring are
explicitly out of scope here — see the Phase 2 task brief.

Slug derivation (stable identity — SCHEMA.md §2 "the path is the identity"): the video's
``source.title`` is slugified (lowercased, non-alnum runs collapsed to single hyphens,
stripped). If that yields nothing usable (empty/untitled source), the slug falls back to the
entry's ``entry_id``. The title is not expected to change once an entry is filed, so the slug
stays stable across re-exports of the same entry.

``published`` is not fetched by Distil today (YouTube oEmbed does not return a publish date),
so it is set to the entry's capture date as the closest available honest proxy; this is
documented here rather than invented silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from .ingest import Transcript
from .models import KBEntry
from .source import _youtube_video_id, display_title

_SLUG_RUN = re.compile(r"[^a-z0-9]+")


def slug_for_entry(entry: KBEntry) -> str:
    """Deterministic, stable OKF slug for ``entry`` (see module docstring)."""
    slug = _slugify(entry.source.title)
    return slug or entry.entry_id


def _slugify(text: str) -> str:
    return _SLUG_RUN.sub("-", text.strip().lower()).strip("-")


def export_entry(entry: KBEntry, transcript: Transcript, okf_root: str | Path) -> None:
    """Write/refresh ``sources/<slug>.md`` + ``raw/<slug>.md`` and regenerate both indexes."""
    root = Path(okf_root)
    sources_dir = root / "sources"
    raw_dir = root / "raw"
    sources_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    slug = slug_for_entry(entry)
    (raw_dir / f"{slug}.md").write_text(_render_raw(entry, transcript, slug), encoding="utf-8")
    (sources_dir / f"{slug}.md").write_text(_render_source(entry, slug), encoding="utf-8")
    _rebuild_indexes(root)


def remove_entry(entry: KBEntry, okf_root: str | Path) -> None:
    """Delete an entry's OKF pages (if present) and regenerate both indexes."""
    root = Path(okf_root)
    slug = slug_for_entry(entry)
    (root / "sources" / f"{slug}.md").unlink(missing_ok=True)
    (root / "raw" / f"{slug}.md").unlink(missing_ok=True)
    _rebuild_indexes(root)


# ---- page rendering ----------------------------------------------------------------------


def _render_source(entry: KBEntry, slug: str) -> str:
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
    return "\n".join(lines)


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


# ---- indexes -------------------------------------------------------------------------------


def _rebuild_indexes(root: Path) -> None:
    sources_dir = root / "sources"
    entries: list[tuple[str, str, str]] = []  # (slug, title, description)
    if sources_dir.exists():
        for path in sorted(sources_dir.glob("*.md")):
            if path.name == "index.md":
                continue
            text = path.read_text(encoding="utf-8")
            title = _frontmatter_field(text, "title") or path.stem
            description = _frontmatter_field(text, "description") or ""
            entries.append((path.stem, title, description))

    sources_index = ["# Sources", ""]
    for slug, title, description in entries:
        sources_index.append(f"- [{title}]({slug}.md) - {description}")
    sources_index.append("")
    (sources_dir).mkdir(parents=True, exist_ok=True)
    (sources_dir / "index.md").write_text("\n".join(sources_index), encoding="utf-8")

    root_index = [
        "---",
        'okf_version: "0.1"',
        "---",
        "",
        "# Distil OKF Bundle",
        "",
        "Per-video OKF layer generated by Distil. Concepts and entities are not yet "
        "synthesized (that begins in a later phase); this bundle currently covers "
        "`sources/` (neutral per-video summaries) and `raw/` (immutable transcripts) only.",
        "",
        "## Sources",
        "",
    ]
    for slug, title, _description in entries:
        root_index.append(f"- [{title}](sources/{slug}.md)")
    root_index.append("")
    root_index.append("See [sources/index.md](sources/index.md) for one-line descriptions.")
    root_index.append("")
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.md").write_text("\n".join(root_index), encoding="utf-8")


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
