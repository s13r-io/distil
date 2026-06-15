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
                entry.source.title,
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

    def list_entries(self) -> list[EntryIndexRow]:
        cur = self._conn.execute("SELECT * FROM entries ORDER BY created_at DESC, entry_id")
        return [self._row_to_index(r) for r in cur.fetchall()]

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
            text = self._item_embed_text(item)
            self._store_vector(entry.entry_id, item.item_id, embedder.embed(text),
                               embedder.model_name)
        self._conn.commit()

    @staticmethod
    def _item_embed_text(item) -> str:
        parts = [item.statement]
        if item.rationale:
            parts.append(item.rationale)
        if item.scope:
            parts.append(item.scope)
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
                    embedder.embed(self._item_embed_text(item)), embedder.model_name
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
        lines.append(f"# {entry.source.title}")
        lines.append("")
        lines.append(
            f"*Verdict:* {entry.triage.verdict} · *Density:* {entry.triage.density} · "
            f"*Captured:* {entry.source.captured_at}"
        )
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
