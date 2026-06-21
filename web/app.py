"""FastAPI app (v0.3): full mobile-first UI per docs/WEB_UI_SPEC.md.

Sections: Ask (streaming, with all-at-once fallback), Add knowledge (non-blocking ingest into a
background job queue), Activity (job statuses), Library (filter/sort), and a parsed Entry page
with inline scoring. Auth (web/auth.py) is unchanged and gates every data route; /health and
/login stay open.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from distil.cli import _make_client, _make_embedder
from distil.graph import link_graph
from distil.ingest import ingest_file, ingest_text
from distil.pipeline import PipelineConfig, run_pipeline
from distil.profile_update import apply_feedback
from distil.query import ask as run_ask
from distil.query import stream_ask
from distil.source import (
    SourceMetadata,
    SourceMetadataError,
    SourceUrlError,
    clean_source_title,
    display_title,
    fetch_youtube_oembed_metadata,
    normalize_youtube_url,
)
from distil.store import Store

from . import auth
from . import jobs as jobsmod

_USER_ID = "owner"
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_STATIC_DIR = Path(__file__).parent / "static"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "distil_uploads"
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE = None


def _db_path() -> str:
    return os.environ.get("DISTIL_DB_PATH", "./data/distil.db")


def _kb_dir() -> str:
    return os.environ.get("DISTIL_KB_DIR", "./kb")


def _store() -> Store:
    return Store(db_path=_db_path(), kb_dir=_kb_dir())


def _default_profile():
    from distil.models import Profile

    return Profile(user_id=_USER_ID)


def _humanize_tag(tag: str) -> str:
    acronyms = {"ai", "api", "cli", "db", "kb", "llm", "ui", "ux"}
    parts = tag.replace("_", " ").replace("-", " ").split()
    words = [part.upper() if part.lower() in acronyms else part.capitalize() for part in parts]
    return " ".join(words)


_TEMPLATES.env.filters["humanize_tag"] = _humanize_tag


def _distill_job(job: jobsmod.Job) -> dict:
    """Worker callback: run the pipeline for one job, return a small result dict.

    Builds fresh Store/client/embedder on the worker thread (no cross-thread sqlite sharing).
    """
    timings: dict[str, float] = {}
    total_start = perf_counter()
    store = _store()
    profile = store.load_profile(_USER_ID) or _default_profile()
    transcript = _time_block(timings, "ingest", lambda: _load_job_transcript(job))
    client = _make_client()
    embedder = _time_block(timings, "embedder", _cached_safe_embedder)
    source_meta = _time_block(timings, "metadata", lambda: _fetch_source_metadata(job.source_url))
    entry = run_pipeline(
        transcript, profile, store, client,
        source_title=source_meta.title or job.title,
        source_url=job.source_url,
        source_channel=source_meta.channel,
        source_channel_url=source_meta.channel_url,
        source_thumbnail_url=source_meta.thumbnail_url,
        source_metadata_provider=source_meta.metadata_provider,
        source_metadata_fetched_at=source_meta.metadata_fetched_at,
        config=PipelineConfig(
            model_version=os.environ.get("DISTIL_MODEL", ""),
            enable_graph=False,
            timing_callback=lambda stage, seconds: timings.__setitem__(stage, seconds),
        ),
        embedder=embedder,
    )
    total = perf_counter() - total_start
    n = len(entry.knowledge_items)
    if n == 0 and entry.triage.verdict == "little_to_extract":
        _emit_timing_log(job, entry.entry_id, jobsmod.STATUS_LOW_VALUE, entry.triage.verdict, n,
                         timings, total)
        return {"status": jobsmod.STATUS_LOW_VALUE, "entry_id": None,
                "summary": "Not much to extract — verdict little_to_extract. Nothing filed. "
                           f"{_format_timings(timings, total)}"}
    graph_scheduled = _schedule_graph_link(entry.entry_id) if entry.tags.topics else False
    graph_note = " · graph updating" if graph_scheduled else ""
    _emit_timing_log(job, entry.entry_id, jobsmod.STATUS_DONE, entry.triage.verdict, n,
                     timings, total)
    return {"status": jobsmod.STATUS_DONE, "entry_id": entry.entry_id,
            "summary": f"kept {n} item{'s' if n != 1 else ''} · verdict {entry.triage.verdict} "
                       f"· {_format_timings(timings, total)}{graph_note}"}


def _load_job_transcript(job: jobsmod.Job):
    if job.kind == "file":
        p = Path(job.payload)
        try:
            return ingest_file(str(p))
        finally:
            p.unlink(missing_ok=True)
    return ingest_text(job.payload)


def _cached_embedder():
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE
    with _EMBEDDER_LOCK:
        if _EMBEDDER_CACHE is None:
            _EMBEDDER_CACHE = _make_embedder()
        return _EMBEDDER_CACHE


def _cached_safe_embedder():
    try:
        return _cached_embedder()
    except Exception:
        return None


def _time_block(timings: dict[str, float], stage: str, fn):
    start = perf_counter()
    try:
        return fn()
    finally:
        timings[stage] = perf_counter() - start


def _format_timings(timings: dict[str, float], total: float) -> str:
    ordered = [
        "ingest", "metadata", "triage", "extract", "normalize", "link", "note",
        "embedder", "file",
    ]
    parts = [
        f"{stage} {timings[stage]:.1f}s"
        for stage in ordered
        if timings.get(stage, 0.0) >= 0.05
    ]
    detail = ", ".join(parts[:6])
    return f"{total:.1f}s" + (f" ({detail})" if detail else "")


def _emit_timing_log(
    job: jobsmod.Job,
    entry_id: str,
    status: str,
    verdict: str,
    item_count: int,
    timings: dict[str, float],
    total: float,
) -> None:
    payload = {
        "job_id": job.job_id,
        "entry_id": entry_id,
        "status": status,
        "verdict": verdict,
        "item_count": item_count,
        "total_seconds": round(total, 3),
        "timings": {stage: round(seconds, 3) for stage, seconds in sorted(timings.items())},
    }
    print("distil_timing " + json.dumps(payload, sort_keys=True), flush=True)


def _schedule_graph_link(entry_id: str) -> bool:
    thread = threading.Thread(
        target=_graph_link_job,
        args=(entry_id,),
        name=f"distil-graph-{entry_id}",
        daemon=True,
    )
    thread.start()
    return True


def _graph_link_job(entry_id: str) -> None:
    store = _store()
    try:
        entry = store.load_entry(entry_id)
        if not entry.tags.topics:
            return
        related = link_graph(entry, store, _make_client())
        if related:
            entry.related_entries = related
            store.file_entry(entry)
    except Exception:
        return


def _fetch_source_metadata(source_url: str | None) -> SourceMetadata:
    if not source_url:
        return SourceMetadata()
    try:
        return fetch_youtube_oembed_metadata(source_url)
    except SourceMetadataError:
        return SourceMetadata()


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

    def _library_template_context() -> dict:
        rows = _store().list_entries()
        entries = [
            {"entry_id": r.entry_id, "title": r.title, "score": r.score,
             "topics": r.topics, "knowledge_types": r.knowledge_types,
             "created_at": r.created_at}
            for r in rows
        ]
        all_tags = sorted({t for r in rows for t in (list(r.topics) + list(r.knowledge_types))})
        tag_options = [{"value": tag, "label": _humanize_tag(tag)} for tag in all_tags]
        return {"entries": entries, "all_tags": tag_options, "entry_count": len(entries)}

    # ---- home / ask ----
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        rows = _store().list_entries()
        return _TEMPLATES.TemplateResponse(
            request, "index.html",
            {"entry_count": len(rows), "has_entries": bool(rows), "active_page": "ask"},
        )

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        return _TEMPLATES.TemplateResponse(
            request, "library.html",
            {**_library_template_context(), "active_page": "library"},
        )

    # ---- ingest (non-blocking) ----
    @app.post("/ingest")
    async def ingest(
        paste: str = Form(default=""),
        source_url: str = Form(default=""),
        file: UploadFile | None = None,
    ):
        store_jobs = jobsmod.JobStore(_db_path())
        try:
            normalized_url = normalize_youtube_url(source_url)
        except SourceUrlError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".srt", ".txt", ".md"}:
                return JSONResponse({"detail": "Unsupported file type"}, status_code=400)
            dest = _UPLOAD_DIR / f"{os.urandom(6).hex()}{suffix}"
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            job = store_jobs.enqueue(
                kind="file",
                title=clean_source_title(file.filename),
                payload=str(dest),
                source_url=normalized_url,
            )
        elif paste.strip():
            job = store_jobs.enqueue(
                kind="paste",
                title="Pasted transcript",
                payload=paste,
                source_url=normalized_url,
            )
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
                         "irrelevant_now"],
             "active_page": "library"},
        )

    @app.get("/entries/{entry_id}/teaching-note.md")
    def teaching_note_markdown(entry_id: str, download: bool = False):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return JSONResponse({"detail": "not found"}, status_code=404)
        entry = store.load_entry(entry_id)
        title = display_title(
            entry.source.title,
            entry.distilled_note.title if entry.distilled_note is not None else None,
        )
        headers = {}
        if download:
            headers["Content-Disposition"] = (
                f'attachment; filename="{_markdown_filename(title)}"'
            )
        return Response(
            Store.teaching_note_markdown(entry),
            media_type="text/markdown; charset=utf-8",
            headers=headers,
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

    @app.post("/entries/{entry_id}/delete")
    def delete_entry(entry_id: str):
        store = _store()
        if not store.delete_entry(entry_id):
            return JSONResponse({"detail": "not found"}, status_code=404)
        return RedirectResponse(url="/library", status_code=303)

    # ---- ask (JSON, all-at-once fallback) ----
    @app.get("/ask")
    def ask(q: str, lookup: bool = False):
        store = _store()
        result = run_ask(q, store, _cached_embedder(), _make_client(), lookup_only=lookup)
        return _ask_payload(result)

    # ---- ask (streaming) ----
    @app.get("/ask/stream")
    def ask_stream(q: str):
        store = _store()
        embedder = _cached_embedder()
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
             "quote": s.quote, "timestamp": s.timestamp, "title": s.entry_title}
            for s in result.sources
        ],
    }


def _markdown_filename(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", title)
    cleaned = re.sub(r"\s+", "-", cleaned).strip("-._ ")
    return f"{(cleaned or 'teaching-note')[:90]}.md"


def _sse(obj: dict) -> str:
    import json as _json

    return f"data: {_json.dumps(obj)}\n\n"


# Module-level app for `uvicorn web.app:app`. Lazy so importing doesn't fail-closed in tests.
def __getattr__(name):  # pragma: no cover
    if name == "app":
        return create_app()
    raise AttributeError(name)
