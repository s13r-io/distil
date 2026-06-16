"""Background distill job queue (WEB_UI_SPEC §8).

Ingest is non-blocking: ``POST /ingest`` inserts a ``queued`` job and returns immediately. A
single in-process worker thread pulls one job at a time and runs the pipeline, so the web
request never waits on the 10-40s LLM work and rate limits are respected by construction.

Restart-safe: jobs are persisted in SQLite. Any job left ``running`` when the process dies is
re-queued on startup (``recover_interrupted``), so a Railway restart resumes rather than
silently dropping work.

Thread-safety: the worker owns its *own* sqlite connection (a fresh :class:`JobStore`), never
sharing the web app's connection across threads.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Job status values (WEB_UI_SPEC §6). "removed" = taken out of the queue before running.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_LOW_VALUE = "low_value"
STATUS_FAILED = "failed"
STATUS_REMOVED = "removed"

_FINISHED = {STATUS_DONE, STATUS_LOW_VALUE, STATUS_REMOVED}
_AUTOCLEAR_AFTER_SECONDS = 24 * 60 * 60  # done/low_value/removed clear after 24h; failed never


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    kind: str  # "paste" | "file"
    title: str
    payload: str  # pasted text, or a stored file path for uploads
    status: str
    entry_id: str | None
    summary: str | None  # e.g. "kept 6 items - verdict rich"
    error: str | None
    created_at: str
    updated_at: str

    def age_seconds(self) -> float:
        try:
            updated = datetime.fromisoformat(self.updated_at)
        except ValueError:
            return 0.0
        return (datetime.now(timezone.utc) - updated).total_seconds()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "entry_id": self.entry_id,
            "summary": self.summary,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    """SQLite-backed job table. Each instance owns its own connection (thread-local use)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL + busy timeout so the web threadpool and the worker thread can use the same DB
        # file concurrently without "database is locked" / I/O errors (WEB_UI_SPEC §8).
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=30, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.OperationalError:
            pass
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id     TEXT PRIMARY KEY,
                kind       TEXT NOT NULL,
                title      TEXT NOT NULL,
                payload    TEXT NOT NULL,
                status     TEXT NOT NULL,
                entry_id   TEXT,
                summary    TEXT,
                error      TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _row(self, r: sqlite3.Row) -> Job:
        return Job(
            job_id=r["job_id"], kind=r["kind"], title=r["title"], payload=r["payload"],
            status=r["status"], entry_id=r["entry_id"], summary=r["summary"],
            error=r["error"], created_at=r["created_at"], updated_at=r["updated_at"],
        )

    def enqueue(self, *, kind: str, title: str, payload: str) -> Job:
        now = _now()
        job = Job(
            job_id=f"j_{uuid.uuid4().hex[:12]}", kind=kind, title=title, payload=payload,
            status=STATUS_QUEUED, entry_id=None, summary=None, error=None,
            created_at=now, updated_at=now,
        )
        self._conn.execute(
            "INSERT INTO jobs (job_id, kind, title, payload, status, entry_id, summary, "
            "error, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.kind, job.title, job.payload, job.status, None, None, None,
             job.created_at, job.updated_at),
        )
        self._conn.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        r = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def list_active(self) -> list[Job]:
        """Jobs to show in Activity, newest first, after applying the 24h auto-clear rule."""
        self.autoclear()
        cur = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        return [self._row(r) for r in cur.fetchall()]

    def claim_next_queued(self) -> Job | None:
        """Atomically move the oldest queued job to running and return it."""
        with self._conn:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (STATUS_QUEUED,),
            ).fetchone()
            if not r:
                return None
            job = self._row(r)
            self._conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE job_id=? AND status=?",
                (STATUS_RUNNING, _now(), job.job_id, STATUS_QUEUED),
            )
        job.status = STATUS_RUNNING
        return job

    def _set_status(self, job_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?", (status, _now(), job_id)
        )
        self._conn.commit()

    def mark_done(self, job_id: str, *, entry_id: str, summary: str) -> None:
        self._conn.execute(
            "UPDATE jobs SET status=?, entry_id=?, summary=?, updated_at=? WHERE job_id=?",
            (STATUS_DONE, entry_id, summary, _now(), job_id),
        )
        self._conn.commit()

    def mark_low_value(self, job_id: str, *, entry_id: str | None, summary: str) -> None:
        self._conn.execute(
            "UPDATE jobs SET status=?, entry_id=?, summary=?, updated_at=? WHERE job_id=?",
            (STATUS_LOW_VALUE, entry_id, summary, _now(), job_id),
        )
        self._conn.commit()

    def mark_failed(self, job_id: str, *, error: str) -> None:
        self._conn.execute(
            "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
            (STATUS_FAILED, error, _now(), job_id),
        )
        self._conn.commit()

    def remove_queued(self, job_id: str) -> bool:
        """Remove a job from the queue — only legal while still queued (WEB_UI_SPEC §6)."""
        r = self._conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not r or r["status"] != STATUS_QUEUED:
            return False
        self._set_status(job_id, STATUS_REMOVED)
        return True

    def retry(self, job_id: str) -> bool:
        """Re-queue a failed job with its original payload (WEB_UI_SPEC §6)."""
        r = self._conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not r or r["status"] != STATUS_FAILED:
            return False
        self._conn.execute(
            "UPDATE jobs SET status=?, error=NULL, updated_at=? WHERE job_id=?",
            (STATUS_QUEUED, _now(), job_id),
        )
        self._conn.commit()
        return True

    def clear(self, scope: str) -> int:
        """Bulk clear. scope='finished' -> done/low_value/removed; scope='failed' -> failed."""
        if scope == "finished":
            statuses = (STATUS_DONE, STATUS_LOW_VALUE, STATUS_REMOVED)
        elif scope == "failed":
            statuses = (STATUS_FAILED,)
        else:
            return 0
        placeholders = ",".join("?" for _ in statuses)
        cur = self._conn.execute(
            f"DELETE FROM jobs WHERE status IN ({placeholders})", statuses
        )
        self._conn.commit()
        return cur.rowcount

    def autoclear(self) -> int:
        """Delete done/low_value/removed rows older than 24h. Failed rows persist forever."""
        cutoff = time.time() - _AUTOCLEAR_AFTER_SECONDS
        removed = 0
        for r in self._conn.execute(
            "SELECT job_id, status, updated_at FROM jobs WHERE status IN (?,?,?)",
            (STATUS_DONE, STATUS_LOW_VALUE, STATUS_REMOVED),
        ).fetchall():
            try:
                updated = datetime.fromisoformat(r["updated_at"]).timestamp()
            except ValueError:
                continue
            if updated < cutoff:
                self._conn.execute("DELETE FROM jobs WHERE job_id=?", (r["job_id"],))
                removed += 1
        if removed:
            self._conn.commit()
        return removed

    def recover_interrupted(self) -> int:
        """Re-queue jobs left 'running' by a crash/restart (WEB_UI_SPEC §8)."""
        cur = self._conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE status=?",
            (STATUS_QUEUED, _now(), STATUS_RUNNING),
        )
        self._conn.commit()
        return cur.rowcount


class Worker:
    """Single background thread: claim queued job -> run distill_fn -> record outcome.

    ``distill_fn(job)`` does the real pipeline work and returns a small result dict
    ``{"status", "entry_id", "summary"}``. It's injected so tests can drive the worker with a
    fake that makes no LLM calls.
    """

    def __init__(
        self,
        db_path: str | Path,
        distill_fn: Callable[[Job], dict],
        *,
        poll_seconds: float = 1.0,
    ):
        self._db_path = db_path
        self._distill_fn = distill_fn
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._store: JobStore | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Recover interrupted jobs on its own connection before the loop begins.
        JobStore(self._db_path).recover_interrupted()
        self._thread = threading.Thread(target=self._run, name="distil-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        self._store = JobStore(self._db_path)  # worker-owned connection
        while not self._stop.is_set():
            job = self._store.claim_next_queued()
            if job is None:
                self._stop.wait(self._poll)
                continue
            self._process(job)

    def process_once(self) -> bool:
        """Synchronous single-step for tests: claim + process one job. Returns True if it ran."""
        store = self._store or JobStore(self._db_path)
        self._store = store
        job = store.claim_next_queued()
        if job is None:
            return False
        self._process(job)
        return True

    def _process(self, job: Job) -> None:
        store = self._store
        assert store is not None
        try:
            result = self._distill_fn(job)
        except Exception as exc:  # any pipeline/LLM failure -> failed + retryable
            store.mark_failed(job.job_id, error=str(exc) or exc.__class__.__name__)
            return
        status = result.get("status")
        if status == STATUS_LOW_VALUE:
            store.mark_low_value(
                job.job_id, entry_id=result.get("entry_id"), summary=result.get("summary", ""),
            )
        elif status == STATUS_DONE:
            store.mark_done(
                job.job_id, entry_id=result.get("entry_id", ""), summary=result.get("summary", ""),
            )
        else:
            store.mark_failed(job.job_id, error=result.get("error", "unknown pipeline result"))
