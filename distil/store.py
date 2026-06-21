"""Persistence for Distil (ARCHITECTURE.md §3, §4; SCHEMA.md §2).

Two stores in one place:

* **KB entries** are filed as markdown in ``kb/<entry_id>.md``. The YAML-ish front matter
  carries the *full* structured :class:`KBEntry` as JSON (so reload is lossless and the file
  is the source of truth); the body below it is a human-readable rendering.
* A **SQLite index** (``entries`` table) mirrors the searchable fields used for browsing and
  graph candidate lookup, plus a ``profiles`` table for the single-user profile.

The vector layer (read layer, Phase 10) lives in the same ``distil.db``. Each knowledge item
is embedded at the **File** stage and stored alongside a companion row keyed back to its
item/entry (SCHEMA §4). The backend is abstracted: ``sqlite-vec`` (a ``vec0`` virtual table)
when the extension loads, otherwise a pure-Python fallback table that stores the vector as
JSON. Both expose the same API, so retrieval logic and its tests don't depend on the
extension being present.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .embed import Embedder
from .models import KBEntry, Profile
from .source import display_title

_FRONT_MATTER_DELIM = "---"


@dataclass
class VectorRow:
    item_id: str
    entry_id: str
    embedding_model: str
    embedded_at: str


@dataclass
class EntryIndexRow:
    entry_id: str
    title: str
    topics: list[str]
    knowledge_types: list[str]
    score: int | None
    created_at: str
    file_path: str


class Store:
    def __init__(self, db_path: str | Path, kb_dir: str | Path):
        self.db_path = Path(db_path)
        self.kb_dir = Path(kb_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._vec_enabled = self._try_load_sqlite_vec()
        self._init_schema()

    def _try_load_sqlite_vec(self) -> bool:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception:
            # Pure-Python fallback (JSON vectors); same API, slower KNN.
            return False

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                entry_id        TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                topics          TEXT NOT NULL,        -- JSON array
                knowledge_types TEXT NOT NULL,        -- JSON array
                score           INTEGER,
                created_at      TEXT NOT NULL,
                file_path       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                data    TEXT NOT NULL                 -- JSON blob
            );
            -- Companion metadata for every embedded item (SCHEMA §4).
            CREATE TABLE IF NOT EXISTS item_vectors_meta (
                item_id         TEXT NOT NULL,
                entry_id        TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedded_at     TEXT NOT NULL,
                vec             TEXT,                 -- JSON vector (fallback backend only)
                PRIMARY KEY (entry_id, item_id)
            );
            """
        )
        self._conn.commit()

    # ---- KB entries -------------------------------------------------------------------

    def entry_path(self, entry_id: str) -> Path:
        return self.kb_dir / f"{entry_id}.md"

    def file_entry(self, entry: KBEntry, *, embedder: Embedder | None = None) -> Path:
        """Write the markdown file, upsert the index row, and (if an embedder is given) embed
        each knowledge item into the vector store. Returns the file path."""
        path = self.entry_path(entry.entry_id)
        path.write_text(self._render_markdown(entry), encoding="utf-8")

        self._conn.execute(
            """
            INSERT INTO entries
                (entry_id, title, topics, knowledge_types, score, created_at, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                title=excluded.title,
                topics=excluded.topics,
                knowledge_types=excluded.knowledge_types,
                score=excluded.score,
                created_at=excluded.created_at,
                file_path=excluded.file_path
            """,
            (
                entry.entry_id,
                display_title(
                    entry.source.title,
                    entry.distilled_note.title if entry.distilled_note is not None else None,
                ),
                json.dumps(entry.tags.topics),
                json.dumps(entry.tags.knowledge_types),
                entry.feedback.score,
                entry.meta.created_at,
                str(path),
            ),
        )
        self._conn.commit()
        if embedder is not None:
            self._embed_entry_items(entry, embedder)
        return path

    def load_entry(self, entry_id: str) -> KBEntry:
        text = self.entry_path(entry_id).read_text(encoding="utf-8")
        payload = self._parse_front_matter(text)
        return KBEntry.model_validate_json(payload)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a KB entry, its SQLite index row, and its item vectors."""
        path = self.entry_path(entry_id)
        file_existed = path.exists()
        path.unlink(missing_ok=True)
        with self._conn:
            self._conn.execute("DELETE FROM item_vectors_meta WHERE entry_id = ?", (entry_id,))
            cur = self._conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        return file_existed or cur.rowcount > 0

    def list_entries(self) -> list[EntryIndexRow]:
        cur = self._conn.execute("SELECT * FROM entries ORDER BY created_at DESC, entry_id")
        rows: list[EntryIndexRow] = []
        stale: list[tuple[str, bool]] = []
        for r in cur.fetchall():
            row = self._row_to_index(r)
            if Path(row.file_path).exists():
                try:
                    entry = self.load_entry(row.entry_id)
                except Exception:
                    rows.append(row)
                    continue
                if entry.triage.verdict == "little_to_extract" and not entry.knowledge_items:
                    stale.append((row.entry_id, True))
                else:
                    rows.append(row)
            else:
                stale.append((row.entry_id, False))
        if stale:
            with self._conn:
                for entry_id, remove_file in stale:
                    if remove_file:
                        self.entry_path(entry_id).unlink(missing_ok=True)
                    self._conn.execute("DELETE FROM item_vectors_meta WHERE entry_id = ?", (entry_id,))
                    self._conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        return rows

    def find_candidates(
        self,
        *,
        topics: list[str],
        knowledge_types: list[str],
        exclude: str | None = None,
    ) -> list[EntryIndexRow]:
        """Deterministic graph candidate lookup: entries sharing any topic/type (no LLM)."""
        results: dict[str, EntryIndexRow] = {}
        wanted_topics = set(topics)
        wanted_types = set(knowledge_types)
        for row in self.list_entries():
            if row.entry_id == exclude:
                continue
            if wanted_topics & set(row.topics) or wanted_types & set(row.knowledge_types):
                results[row.entry_id] = row
        return list(results.values())

    # ---- Vector store (read layer) ----------------------------------------------------

    def _embed_entry_items(self, entry: KBEntry, embedder: Embedder) -> None:
        for item in entry.knowledge_items:
            text = self._item_embed_text(item, entry)
            self._store_vector(entry.entry_id, item.item_id, embedder.embed(text),
                               embedder.model_name)
        self._conn.commit()

    @staticmethod
    def _item_embed_text(item, entry: KBEntry | None = None) -> str:
        parts = [item.statement]
        if item.rationale:
            parts.append(item.rationale)
        if item.scope:
            parts.append(item.scope)
        if entry is not None:
            context = Store.note_context_for_item(entry, item.item_id)
            if context:
                parts.append(context)
        return " ".join(parts)

    @staticmethod
    def note_context_for_item(entry: KBEntry, item_id: str) -> str:
        """Return synthesized-note context that cites this item, for retrieval prompts/vectors."""
        note = entry.distilled_note
        if note is None:
            return ""
        parts: list[str] = []

        def add(text: str, ids: list[str]) -> None:
            text = text.strip()
            if item_id in ids and text and text not in parts:
                parts.append(text)

        add(note.core_takeaway.text, note.core_takeaway.item_ids)
        for section in note.key_points:
            add(section.text, section.item_ids)
        for section in note.why_it_matters:
            add(section.text, section.item_ids)
        for section in note.how_to_apply:
            add(section.text, section.item_ids)
        for section in note.caveats:
            add(section.text, section.item_ids)
        for section in note.review_questions:
            add(section.question, section.item_ids)
        return " ".join(parts)

    def _store_vector(self, entry_id: str, item_id: str, vec: list[float], model: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO item_vectors_meta (item_id, entry_id, embedding_model, embedded_at, vec)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entry_id, item_id) DO UPDATE SET
                embedding_model=excluded.embedding_model,
                embedded_at=excluded.embedded_at,
                vec=excluded.vec
            """,
            (item_id, entry_id, model, now, json.dumps(vec)),
        )

    def vector_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM item_vectors_meta")
        return cur.fetchone()["n"]

    def all_vector_rows(self) -> list[VectorRow]:
        cur = self._conn.execute(
            "SELECT item_id, entry_id, embedding_model, embedded_at FROM item_vectors_meta"
        )
        return [
            VectorRow(r["item_id"], r["entry_id"], r["embedding_model"], r["embedded_at"])
            for r in cur.fetchall()
        ]

    def reindex(self, embedder: Embedder) -> int:
        """Backfill/refresh vectors. Embeds items lacking a current vector for this model.
        Idempotent: items already embedded with the same model are skipped. Returns the count
        of (re)embedded items (T-X2)."""
        existing: dict[tuple[str, str], str] = {}
        for r in self._conn.execute(
            "SELECT entry_id, item_id, embedding_model FROM item_vectors_meta"
        ):
            existing[(r["entry_id"], r["item_id"])] = r["embedding_model"]

        embedded = 0
        for row in self.list_entries():
            entry = self.load_entry(row.entry_id)
            for item in entry.knowledge_items:
                key = (entry.entry_id, item.item_id)
                if existing.get(key) == embedder.model_name:
                    continue  # already current
                self._store_vector(
                    entry.entry_id, item.item_id,
                    embedder.embed(self._item_embed_text(item, entry)), embedder.model_name
                )
                embedded += 1
        self._conn.commit()
        return embedded

    def iter_item_vectors(self):
        """Yield (item_id, entry_id, vector) for every embedded item (retrieval reads this)."""
        for r in self._conn.execute("SELECT item_id, entry_id, vec FROM item_vectors_meta"):
            if r["vec"] is not None:
                yield r["item_id"], r["entry_id"], json.loads(r["vec"])

    # ---- Profile ----------------------------------------------------------------------

    def save_profile(self, profile: Profile) -> None:
        self._conn.execute(
            "INSERT INTO profiles (user_id, data) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data",
            (profile.user_id, profile.model_dump_json()),
        )
        self._conn.commit()

    def load_profile(self, user_id: str) -> Profile | None:
        cur = self._conn.execute("SELECT data FROM profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return Profile.model_validate_json(row["data"])

    # ---- rendering / parsing ----------------------------------------------------------

    @staticmethod
    def _render_markdown(entry: KBEntry) -> str:
        """Front matter = full entry JSON (lossless); body = readable rendering."""
        front = entry.model_dump_json(indent=2)
        lines = [_FRONT_MATTER_DELIM, front, _FRONT_MATTER_DELIM, ""]
        if entry.distilled_note is not None:
            return "\n".join(lines + Store._render_note_body(entry))

        lines.append(f"# {entry.source.title}")
        lines.append("")
        lines.append(
            f"*Verdict:* {entry.triage.verdict} · *Density:* {entry.triage.density} · "
            f"*Captured:* {entry.source.captured_at}"
        )
        if entry.source.url:
            lines.append("")
            lines.append(f"*Source:* [Watch on YouTube]({entry.source.url})")
        Store._append_source_metadata(lines, entry)
        lines.append("")
        if entry.knowledge_items:
            lines.append("## Knowledge")
            lines.append("")
            for item in entry.knowledge_items:
                ts = f" ({item.provenance.timestamp})" if item.provenance.timestamp else ""
                lines.append(f"- **[{item.type}]** {item.statement}")
                lines.append(f"  > “{item.provenance.quote}”{ts}")
            lines.append("")
        if entry.application_links:
            lines.append("## Apply it")
            lines.append("")
            for link in entry.application_links:
                tag = " _(novelty)_" if link.novelty_flag else ""
                lines.append(
                    f"- _{link.application_form}_ → {link.scenario} "
                    f"(goal `{link.linked_goal_id}`){tag}"
                )
            lines.append("")
        if entry.related_entries:
            lines.append("## Related")
            lines.append("")
            for rel in entry.related_entries:
                lines.append(f"- {rel.relation} → `{rel.target}`")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def teaching_note_markdown(entry: KBEntry) -> str:
        """Reader-facing markdown export for copy/download, without front matter or item IDs."""
        if entry.distilled_note is None:
            text = Store._render_markdown(entry)
            try:
                return text.split(f"{_FRONT_MATTER_DELIM}\n", 2)[-1].strip() + "\n"
            except ValueError:
                return text.strip() + "\n"

        note = entry.distilled_note
        lines: list[str] = [f"# {display_title(entry.source.title, note.title)}", ""]
        lines.append("## Metadata")
        lines.append("")
        if entry.source.title:
            lines.append(f"- Source title: {entry.source.title}")
        if entry.source.url:
            lines.append(f"- Source URL: {entry.source.url}")
        if entry.source.channel:
            lines.append(f"- Channel: {entry.source.channel}")
        lines.append(f"- Captured: {entry.source.captured_at}")
        lines.append(f"- Verdict: {entry.triage.verdict}")
        lines.append(f"- Density: {entry.triage.density}")
        if note.topics:
            lines.append("- Tags: " + ", ".join(Store._display_tag(topic) for topic in note.topics))
        lines.append("")

        lines.append("## Core takeaway")
        lines.append("")
        lines.append(note.core_takeaway.text)
        lines.append("")
        Store._append_export_section(lines, "Key points", [p.text for p in note.key_points])
        Store._append_export_section(lines, "Why it matters", [p.text for p in note.why_it_matters])
        Store._append_export_section(
            lines, "How to apply this", [step.text for step in note.how_to_apply], ordered=True
        )
        Store._append_export_section(lines, "Caveats", [c.text for c in note.caveats])
        Store._append_export_section(
            lines,
            "Review questions",
            [question.question for question in note.review_questions],
            ordered=True,
        )
        if note.topics:
            lines.append("## Tags")
            lines.append("")
            for topic in note.topics:
                lines.append(f"- {Store._display_tag(topic)}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_note_body(entry: KBEntry) -> list[str]:
        note = entry.distilled_note
        assert note is not None
        lines: list[str] = []
        lines.append(f"# {display_title(entry.source.title, note.title)}")
        lines.append("")
        lines.append(
            f"*Verdict:* {entry.triage.verdict} · *Density:* {entry.triage.density} · "
            f"*Captured:* {entry.source.captured_at}"
        )
        if entry.source.url:
            lines.append("")
            lines.append(f"*Source:* [Watch on YouTube]({entry.source.url})")
        Store._append_source_metadata(lines, entry)
        if note.topics:
            lines.append("")
            lines.append("*Tags:* " + ", ".join(Store._display_tag(topic) for topic in note.topics))
        lines.append("")

        lines.append("## Core takeaway")
        lines.append("")
        lines.append(Store._with_refs(note.core_takeaway.text, note.core_takeaway.item_ids))
        lines.append("")

        Store._append_grounded_section(lines, "Key points", note.key_points)
        Store._append_grounded_section(lines, "Why it matters", note.why_it_matters)
        if note.how_to_apply:
            lines.append("## How to apply this")
            lines.append("")
            for step in note.how_to_apply:
                lines.append(f"- {Store._with_refs(step.text, step.item_ids)}")
            lines.append("")
        Store._append_grounded_section(lines, "Caveats", note.caveats)
        if note.review_questions:
            lines.append("## Review questions")
            lines.append("")
            for question in note.review_questions:
                lines.append(f"- {Store._with_refs(question.question, question.item_ids)}")
            lines.append("")

        if entry.knowledge_items:
            lines.append("<details>")
            lines.append("<summary>Source evidence</summary>")
            lines.append("")
            for item in entry.knowledge_items:
                ts = f" ({item.provenance.timestamp})" if item.provenance.timestamp else ""
                lines.append(f"- **{item.item_id} [{item.type}]** {item.statement}")
                lines.append(f"  > \"{item.provenance.quote}\"{ts}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if entry.related_entries:
            lines.append("## Related")
            lines.append("")
            for rel in entry.related_entries:
                lines.append(f"- {rel.relation} -> `{rel.target}`")
            lines.append("")
        return lines

    @staticmethod
    def _append_grounded_section(lines: list[str], title: str, sections) -> None:
        if not sections:
            return
        lines.append(f"## {title}")
        lines.append("")
        for section in sections:
            lines.append(f"- {Store._with_refs(section.text, section.item_ids)}")
        lines.append("")

    @staticmethod
    def _with_refs(text: str, item_ids: list[str]) -> str:
        if not item_ids:
            return text
        refs = ", ".join(f"`{item_id}`" for item_id in item_ids)
        return f"{text} ({refs})"

    @staticmethod
    def _display_tag(tag: str) -> str:
        acronyms = {"ai", "api", "cli", "db", "kb", "llm", "ui", "ux"}
        words = []
        for part in tag.replace("_", " ").replace("-", " ").split():
            words.append(part.upper() if part.lower() in acronyms else part.capitalize())
        return " ".join(words)

    @staticmethod
    def _append_export_section(
        lines: list[str], title: str, values: list[str], *, ordered: bool = False
    ) -> None:
        values = [value.strip() for value in values if value.strip()]
        if not values:
            return
        lines.append(f"## {title}")
        lines.append("")
        for idx, value in enumerate(values, start=1):
            prefix = f"{idx}." if ordered else "-"
            lines.append(f"{prefix} {value}")
        lines.append("")

    @staticmethod
    def _append_source_metadata(lines: list[str], entry: KBEntry) -> None:
        if entry.source.title:
            lines.append(f"*Video:* {entry.source.title}")
        if entry.source.channel:
            channel = entry.source.channel
            if entry.source.channel_url:
                channel = f"[{channel}]({entry.source.channel_url})"
            lines.append(f"*Channel:* {channel}")

    @staticmethod
    def _parse_front_matter(text: str) -> str:
        if not text.startswith(_FRONT_MATTER_DELIM):
            raise ValueError("KB entry file is missing front matter")
        rest = text[len(_FRONT_MATTER_DELIM) :].lstrip("\n")
        end = rest.find(f"\n{_FRONT_MATTER_DELIM}")
        if end == -1:
            raise ValueError("KB entry front matter is not terminated")
        return rest[:end]

    @staticmethod
    def _row_to_index(r: sqlite3.Row) -> EntryIndexRow:
        return EntryIndexRow(
            entry_id=r["entry_id"],
            title=r["title"],
            topics=json.loads(r["topics"]),
            knowledge_types=json.loads(r["knowledge_types"]),
            score=r["score"],
            created_at=r["created_at"],
            file_path=r["file_path"],
        )
