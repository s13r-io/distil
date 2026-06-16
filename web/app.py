"""FastAPI app (v0.3): full mobile-first UI per docs/WEB_UI_SPEC.md.

Sections: Ask (streaming, with all-at-once fallback), Add knowledge (non-blocking ingest into a
background job queue), Activity (job statuses), Library (filter/sort), and a parsed Entry page
with inline scoring. Auth (web/auth.py) is unchanged and gates every data route; /health and
/login stay open.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from distil.cli import _make_client, _make_embedder, _safe_embedder
from distil.ingest import ingest_file, ingest_text
from distil.pipeline import PipelineConfig, run_pipeline
from distil.profile_update import apply_feedback
from distil.query import ask as run_ask
from distil.query import stream_ask
from distil.store import Store

from . import auth
from . import jobs as jobsmod

_USER_ID = "owner"
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_STATIC_DIR = Path(__file__).parent / "static"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "distil_uploads"


def _db_path() -> str:
    return os.environ.get("DISTIL_DB_PATH", "./data/distil.db")


def _kb_dir() -> str:
    return os.environ.get("DISTIL_KB_DIR", "./kb")


def _store() -> Store:
    return Store(db_path=_db_path(), kb_dir=_kb_dir())


def _default_profile():
    from distil.models import Profile

    return Profile(user_id=_USER_ID)


def _distill_job(job: jobsmod.Job) -> dict:
    """Worker callback: run the pipeline for one job, return a small result dict.

    Builds fresh Store/client/embedder on the worker thread (no cross-thread sqlite sharing).
    """
    store = _store()
    profile = store.load_profile(_USER_ID) or _default_profile()
    if job.kind == "file":
        p = Path(job.payload)
        try:
            transcript = ingest_file(str(p))
        finally:
            p.unlink(missing_ok=True)
    else:
        transcript = ingest_text(job.payload)
    client = _make_client()
    embedder = _safe_embedder()
    entry = run_pipeline(
        transcript, profile, store, client,
        source_title=job.title,
        config=PipelineConfig(model_version=os.environ.get("DISTIL_MODEL", "")),
        embedder=embedder,
    )
    n = len(entry.knowledge_items)
    if n == 0 and entry.triage.verdict == "little_to_extract":
        return {"status": jobsmod.STATUS_LOW_VALUE, "entry_id": entry.entry_id,
                "summary": "Not much to extract — verdict little_to_extract. Nothing filed."}
    return {"status": jobsmod.STATUS_DONE, "entry_id": entry.entry_id,
            "summary": f"kept {n} item{'s' if n != 1 else ''} · verdict {entry.triage.verdict}"}


def create_app() -> FastAPI:
    auth.assert_startup_safe()  # fail closed before serving (T-A1)
    app = FastAPI(title="Distil", docs_url=None, redoc_url=None)
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    worker = jobsmod.Worker(_db_path(), _distill_job)

    @app.on_event("startup")
    def _start_worker():
        worker.start()

    @app.on_event("shutdown")
    def _stop_worker():
        worker.stop()

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        if not auth.path_is_open(request.url.path) and not request.url.path.startswith("/static"):
            if not auth.request_is_authorized(request):
                accepts_html = "text/html" in request.headers.get("accept", "")
                if accepts_html:
                    return RedirectResponse(url="/login", status_code=303)
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    # ---- open routes ----
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    def login_get():
        return auth.login_page()

    @app.post("/login")
    def login_post(secret: str = Form(...)):
        return auth.login_response(secret)

    @app.get("/logout")
    def logout():
        return auth.logout_response()

    # ---- home ----
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        rows = _store().list_entries()
        entries = [
            {"entry_id": r.entry_id, "title": r.title, "score": r.score,
             "topics": r.topics, "knowledge_types": r.knowledge_types,
             "created_at": r.created_at}
            for r in rows
        ]
        all_tags = sorted({t for r in rows for t in (list(r.topics) + list(r.knowledge_types))})
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"entries": entries, "all_tags": all_tags},
        )

    # ---- ingest (non-blocking) ----
    @app.post("/ingest")
    async def ingest(
        paste: str = Form(default=""),
        file: UploadFile | None = None,
    ):
        store_jobs = jobsmod.JobStore(_db_path())
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".srt", ".txt", ".md"}:
                return JSONResponse({"detail": "Unsupported file type"}, status_code=400)
            dest = _UPLOAD_DIR / f"{os.urandom(6).hex()}{suffix}"
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            job = store_jobs.enqueue(kind="file", title=file.filename, payload=str(dest))
        elif paste.strip():
            from datetime import datetime

            title = f"Pasted transcript · {datetime.now().strftime('%H:%M')}"
            job = store_jobs.enqueue(kind="paste", title=title, payload=paste)
        else:
            return JSONResponse({"detail": "Nothing to distil"}, status_code=400)
        return {"job_id": job.job_id, "status": job.status}

    # ---- jobs (Activity) ----
    @app.get("/jobs")
    def jobs_list():
        return [j.to_dict() for j in jobsmod.JobStore(_db_path()).list_active()]

    @app.post("/jobs/{job_id}/remove")
    def jobs_remove(job_id: str):
        ok = jobsmod.JobStore(_db_path()).remove_queued(job_id)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

    @app.post("/jobs/{job_id}/retry")
    def jobs_retry(job_id: str):
        ok = jobsmod.JobStore(_db_path()).retry(job_id)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

    @app.post("/jobs/clear")
    def jobs_clear(scope: str = "finished"):
        n = jobsmod.JobStore(_db_path()).clear(scope)
        return {"cleared": n}

    # ---- entries ----
    @app.get("/entries")
    def entries():
        return [
            {"entry_id": r.entry_id, "title": r.title, "score": r.score}
            for r in _store().list_entries()
        ]

    @app.get("/entries/{entry_id}", response_class=HTMLResponse)
    def entry_page(request: Request, entry_id: str):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return HTMLResponse("<p>Entry not found.</p>", status_code=404)
        e = store.load_entry(entry_id)
        mix = [(s.type, round(s.share * 100)) for s in e.triage.knowledge_types_present]
        return _TEMPLATES.TemplateResponse(
            request, "entry.html",
            {"e": e, "mix": mix,
             "reasons": ["relevant", "already_knew", "bad_source", "wrong_for_me",
                         "irrelevant_now"]},
        )

    @app.post("/entries/{entry_id}/score")
    def score(entry_id: str, score: int = Form(...), reason: str = Form(...)):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return JSONResponse({"detail": "not found"}, status_code=404)
        e = store.load_entry(entry_id)
        e.feedback.score = score
        try:
            e.feedback.reason = reason
            e = e.model_validate(e.model_dump())
        except Exception:
            return JSONResponse({"detail": "invalid reason"}, status_code=400)
        store.file_entry(e)
        profile = store.load_profile(_USER_ID) or _default_profile()
        store.save_profile(apply_feedback(profile, e))
        return {"ok": True, "score": score, "reason": reason}

    # ---- ask (JSON, all-at-once fallback) ----
    @app.get("/ask")
    def ask(q: str, lookup: bool = False):
        store = _store()
        result = run_ask(q, store, _make_embedder(), _make_client(), lookup_only=lookup)
        return _ask_payload(result)

    # ---- ask (streaming) ----
    @app.get("/ask/stream")
    def ask_stream(q: str):
        store = _store()
        embedder = _make_embedder()
        client = _make_client()

        def gen():

            try:
                for ev in stream_ask(q, store, embedder, client):
                    if ev.kind == "delta":
                        yield _sse({"type": "delta", "text": ev.text})
                    elif ev.kind == "abstain":
                        yield _sse({"type": "abstain", "message": ev.text})
                    elif ev.kind == "error":
                        yield _sse({"type": "error", "message": ev.text})
                    elif ev.kind == "final":
                        yield _sse({"type": "final", **_ask_payload(ev.result)})
            except Exception as exc:  # last-resort guard → client shows retry
                yield _sse({"type": "error", "message": str(exc) or "stream failed"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _ask_payload(result) -> dict:
    return {
        "abstained": result.abstained,
        "message": result.message,
        "answer": result.answer,
        "conflict": result.conflict,
        "sources": [
            {"entry_id": s.entry_id, "item_id": s.item_id,
             "quote": s.quote, "timestamp": s.timestamp}
            for s in result.sources
        ],
    }


def _sse(obj: dict) -> str:
    import json as _json

    return f"data: {_json.dumps(obj)}\n\n"


# Module-level app for `uvicorn web.app:app`. Lazy so importing doesn't fail-closed in tests.
def __getattr__(name):  # pragma: no cover
    if name == "app":
        return create_app()
    raise AttributeError(name)
