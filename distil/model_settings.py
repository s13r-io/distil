"""Durable per-stage model overrides — the storage half of the model-settings surface
(``/settings`` in the web app). Stored beside the database on the same mounted volume
(``DISTIL_DB_PATH``) so a choice made from inside the app survives a restart/redeploy with no
env var and no visit to the hosting provider.

Precedence, enforced by :func:`distil.model_config.resolve_stage_model`: a stored setting here
overrides ``DISTIL_MODEL_<STAGE>``, which overrides the built-in tier default. A pipeline run
already holding an already-constructed :class:`~distil.llm.AnthropicClient` keeps whatever model
that client was built with — the model string is fixed at construction and never re-resolved
mid-call — so a change here takes effect starting with the next ``make_stage_client`` call, i.e.
the next video, never retroactively.

This module never stores or accepts a credential: the ``model_settings`` table has exactly one
content column (``model``, a plain model-id string). ``ANTHROPIC_API_KEY`` is untouched and
lives only in the environment, as always.

:class:`ModelSettingsStore` opens its own direct sqlite connection to ``DISTIL_DB_PATH`` rather
than going through :class:`distil.store.Store` — that keeps this module's dependency on the
database to "a file path", so :func:`distil.model_config.resolve_stage_model` (a pure,
stateless-until-now function used everywhere, including at import time in tests with no
database configured) can consult it without depending on ``Store``'s heavier constructor
(kb_dir, OKF export, sqlite-vec loading, ...). ``model_config.py`` imports this module lazily
(inside the functions that need it) specifically to avoid a cycle: this module imports
:data:`distil.model_config.MODEL_MAX_OUTPUT_TOKENS` at module level.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .model_config import MODEL_MAX_OUTPUT_TOKENS


class UnknownModelError(ValueError):
    """Raised when a stage is set to a model string Distil doesn't recognize — refused at the
    point of setting it, never silently accepted."""


# The provider's current + still-active model catalogue (Anthropic's own model list, not
# guessed — see the PR that introduced this settings surface for the source), unioned with
# every model this codebase already has ceiling data for (MODEL_MAX_OUTPUT_TOKENS) so the two
# sources named in the brief ("MODEL_MAX_OUTPUT_TOKENS and the provider's catalogue") are both
# actually consulted. Retired models (no longer callable at all) are deliberately excluded —
# accepting one here would just move today's failure from "at the point of setting it" to "at
# the next video", the exact footgun this validation exists to prevent.
KNOWN_MODELS = frozenset(
    {
        # Current
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        # Legacy (still active)
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-5",
        # Deprecated (still active; retiring on a published date)
        "claude-sonnet-4-0",
        "claude-opus-4-0",
        "claude-3-haiku-20240307",
    }
    | set(MODEL_MAX_OUTPUT_TOKENS)
)


def is_known_model(model: str) -> bool:
    return model in KNOWN_MODELS


def validate_model_string(model: str) -> None:
    """Raise :class:`UnknownModelError` with a clear reason for a model string this codebase
    doesn't recognize. Called from :meth:`ModelSettingsStore.set` — the point of setting a
    stage's model, not the next time it's resolved."""
    if not is_known_model(model):
        raise UnknownModelError(
            f"'{model}' is not a model Distil recognizes. Known models: "
            f"{', '.join(sorted(KNOWN_MODELS))}."
        )


def _default_db_path() -> str:
    return os.environ.get("DISTIL_DB_PATH", "./data/distil.db")


@dataclass(frozen=True)
class StoredModelSetting:
    stage: str
    model: str
    updated_at: str


class ModelSettingsStore:
    """A dedicated ``model_settings`` table in the same sqlite file :class:`distil.store.Store`
    and :class:`web.jobs.JobStore` already use (``DISTIL_DB_PATH``) — its own connection, not a
    shared one, so this module has no constructor-time dependency on either of those classes."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else Path(_default_db_path())

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        """``create=False`` (every read path) never creates the db file or its parent
        directory — a fresh install with no database yet must resolve "no stored setting"
        without any filesystem side effect."""
        if not create and not self.db_path.exists():
            return None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS model_settings ("
            "stage TEXT PRIMARY KEY, model TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        return conn

    def get(self, stage: str) -> str | None:
        conn = self._connect(create=False)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT model FROM model_settings WHERE stage = ?", (stage,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def all(self) -> dict[str, StoredModelSetting]:
        conn = self._connect(create=False)
        if conn is None:
            return {}
        try:
            rows = conn.execute("SELECT stage, model, updated_at FROM model_settings").fetchall()
            return {r[0]: StoredModelSetting(stage=r[0], model=r[1], updated_at=r[2]) for r in rows}
        finally:
            conn.close()

    def set(self, stage: str, model: str) -> None:
        """Store ``model`` as ``stage``'s override. Raises :class:`UnknownModelError` — and
        writes nothing — for a model string outside :data:`KNOWN_MODELS`."""
        model = model.strip()
        validate_model_string(model)
        conn = self._connect(create=True)
        assert conn is not None
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute(
                    "INSERT INTO model_settings (stage, model, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(stage) DO UPDATE SET "
                    "model=excluded.model, updated_at=excluded.updated_at",
                    (stage, model, now),
                )
        finally:
            conn.close()

    def clear(self, stage: str) -> None:
        """Remove ``stage``'s stored override, reverting it to whatever
        ``DISTIL_MODEL_<STAGE>``/the tier default resolves to. A no-op (not an error) when
        nothing was stored — this is the one-obvious-action revert path, safe to click twice."""
        conn = self._connect(create=False)
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM model_settings WHERE stage = ?", (stage,))
        finally:
            conn.close()
