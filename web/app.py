"""FastAPI app (v0.3): full mobile-first UI per docs/WEB_UI_SPEC.md.

Sections: Ask (streaming, with all-at-once fallback), Add knowledge (non-blocking ingest into a
background job queue), Activity (job statuses), Library (filter/sort), and a parsed Entry page
with inline scoring. Auth (web/auth.py) is unchanged and gates every data route; /health and
/login stay open.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
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

from distil import okf, youtube
from distil.canonicalize import run_delete_entry_stage
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
from distil.synthesize_concept import find_claim_contradictions
from distil.youtube import YoutubeFetchError

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
    if job.kind == "youtube":
        return youtube.fetch_video_transcript(job.payload)
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


def _enqueue_youtube_source(store_jobs: jobsmod.JobStore, url: str):
    """ADD input: a bare YouTube video or playlist URL, no paste/file.

    A playlist enqueues one ``youtube`` job per video through the existing single-worker
    queue; a bad video's caption/availability failure only fails its own job (jobs.py
    ``Worker`` already isolates per-job exceptions), so one skipped video never blocks the
    rest of the playlist.
    """
    if youtube.is_playlist_url(url):
        try:
            video_urls = youtube.list_playlist_video_urls(url)
        except YoutubeFetchError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except subprocess.TimeoutExpired:
            return JSONResponse({"detail": "Listing the playlist timed out."}, status_code=400)
        except OSError as exc:
            return JSONResponse({"detail": f"Could not run yt-dlp: {exc}"}, status_code=400)
        job_ids = [
            store_jobs.enqueue(
                kind="youtube", title="YouTube video", payload=v_url, source_url=v_url,
            ).job_id
            for v_url in video_urls
        ]
        return {"status": "queued", "job_ids": job_ids, "count": len(job_ids)}
    try:
        normalized_url = normalize_youtube_url(url)
    except SourceUrlError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    job = store_jobs.enqueue(
        kind="youtube", title="YouTube video", payload=normalized_url, source_url=normalized_url,
    )
    return {"job_id": job.job_id, "status": job.status}


def _fetch_source_metadata(source_url: str | None) -> SourceMetadata:
    if not source_url:
        return SourceMetadata()
    try:
        return fetch_youtube_oembed_metadata(source_url)
    except SourceMetadataError:
        return SourceMetadata()


def _concept_detail_context(store: Store, concept) -> dict:
    """Template context for a concept detail page: claims with resolved citations/
    contradictions, deduped sources, and typed edges grouped by relation. Every citation and
    source link resolves from verified member/entry data, never model text — same discipline
    ``okf._render_concept`` already applies to the OKF page for this concept."""
    member_entries: dict[str, object] = {}
    for member in concept.members:
        if member.entry_id in member_entries:
            continue
        try:
            member_entries[member.entry_id] = store.load_entry(member.entry_id)
        except Exception:
            continue

    members_by_item = {m.item_id: m for m in concept.members}
    contradictions = find_claim_contradictions(concept, store)
    claims = []
    for idx, claim in enumerate(concept.claims):
        citations = []
        for item_id in claim.item_ids:
            member = members_by_item.get(item_id)
            if member is None:
                continue
            entry = member_entries.get(member.entry_id)
            citations.append({
                "entry_id": member.entry_id,
                "entry_title": entry.source.title if entry is not None else member.entry_id,
                "item_id": item_id,
                "timestamp": member.timestamp,
            })
        contradiction = None
        rows = contradictions.get(idx)
        if rows:
            contradiction = [
                {
                    "entry_title": (
                        member_entries[entry_id].source.title
                        if entry_id in member_entries else entry_id
                    ),
                    "stance": stance,
                }
                for entry_id, _item_id, stance in rows
            ]
        claims.append({"text": claim.text, "citations": citations, "contradiction": contradiction})

    sources = []
    seen_entries: set[str] = set()
    for member in concept.members:
        if member.entry_id in seen_entries:
            continue
        seen_entries.add(member.entry_id)
        entry = member_entries.get(member.entry_id)
        if entry is None:
            continue
        sources.append({
            "entry_id": member.entry_id,
            "title": entry.source.title,
            "quote": member.quote,
            "timestamp": member.timestamp,
        })

    other_titles = {
        c.concept_id: c.title for c in store.list_concepts() if c.concept_id != concept.concept_id
    }
    edges_by_relation: dict[str, list[dict]] = {"contrasts_with": [], "builds_on": [], "related": []}
    for edge in concept.edges:
        if edge.target_concept_id in other_titles:
            edges_by_relation[edge.relation].append({
                "target_concept_id": edge.target_concept_id,
                "title": other_titles[edge.target_concept_id],
            })

    return {
        "concept": concept,
        "claims": claims,
        "sources": sources,
        "edges_by_relation": edges_by_relation,
        "has_contradiction": any(c["contradiction"] for c in claims),
    }


def _concepts_for_entry(store: Store, entry_id: str) -> list[dict]:
    return [
        {"concept_id": c.concept_id, "title": c.title}
        for c in store.list_concepts()
        if any(m.entry_id == entry_id for m in c.members)
    ]


class _ZipChunkBuffer(io.RawIOBase):
    """A write-only, non-seekable buffer ``zipfile.ZipFile`` can write into; ``get()`` drains
    whatever has accumulated so a caller can yield it as one chunk of a streamed response,
    instead of the whole archive being built in memory before anything is sent."""

    def __init__(self) -> None:
        self._data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self._data += b
        return len(b)

    def get(self) -> bytes:
        chunk = bytes(self._data)
        self._data.clear()
        return chunk


def _iter_bundle_zip(okf_root: Path):
    buf = _ZipChunkBuffer()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if okf_root.exists():
            resolved_root = okf_root.resolve()
            for path in sorted(okf_root.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    path.resolve().relative_to(resolved_root)
                except ValueError:
                    continue  # never include anything outside the bundle directory
                zf.write(path, arcname=str(path.relative_to(okf_root)))
                chunk = buf.get()
                if chunk:
                    yield chunk
    tail = buf.get()
    if tail:
        yield tail


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

    def _concepts_template_context() -> dict:
        concepts = [
            {
                "concept_id": c.concept_id,
                "title": c.title,
                "description": c.description,
                "video_count": len({m.entry_id for m in c.members}),
            }
            for c in _store().list_concepts()
        ]
        return {"concepts": concepts, "concept_count": len(concepts)}

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

    # ---- concepts ----
    @app.get("/concepts", response_class=HTMLResponse)
    def concepts_list(request: Request):
        return _TEMPLATES.TemplateResponse(
            request, "concepts.html",
            {**_concepts_template_context(), "active_page": "concepts"},
        )

    @app.get("/concepts/{concept_id}", response_class=HTMLResponse)
    def concept_page(request: Request, concept_id: str):
        store = _store()
        concept = store.load_concept(concept_id)
        if concept is None:
            return HTMLResponse("<p>Concept not found.</p>", status_code=404)
        return _TEMPLATES.TemplateResponse(
            request, "concept.html",
            {**_concept_detail_context(store, concept), "active_page": "concepts"},
        )

    # ---- ingest (non-blocking) ----
    @app.post("/ingest")
    async def ingest(
        paste: str = Form(default=""),
        source_url: str = Form(default=""),
        file: UploadFile | None = None,
    ):
        store_jobs = jobsmod.JobStore(_db_path())
        has_content = bool(paste.strip()) or (file is not None and bool(file.filename))
        if not has_content and source_url.strip():
            return await asyncio.to_thread(_enqueue_youtube_source, store_jobs, source_url.strip())
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
        slug = okf.slug_for_entry(e, store.okf_root)
        has_transcript = (store.okf_root / "raw" / f"{slug}.md").exists()
        return _TEMPLATES.TemplateResponse(
            request, "entry.html",
            {"e": e, "mix": mix,
             "has_transcript": has_transcript,
             "concepts_for_entry": _concepts_for_entry(store, entry_id),
             "reasons": ["relevant", "already_knew", "bad_source", "wrong_for_me",
                         "irrelevant_now"],
             "active_page": "library"},
        )

    @app.get("/entries/{entry_id}/transcript.md")
    def transcript_markdown(entry_id: str, download: bool = False):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return JSONResponse({"detail": "not found"}, status_code=404)
        entry = store.load_entry(entry_id)
        slug = okf.slug_for_entry(entry, store.okf_root)
        raw_path = store.okf_root / "raw" / f"{slug}.md"
        if not raw_path.exists():
            return JSONResponse({"detail": "transcript not available"}, status_code=404)
        headers = {}
        if download:
            title = display_title(
                entry.source.title,
                entry.distilled_note.title if entry.distilled_note is not None else None,
            )
            filename = _markdown_filename(f"{title}-transcript")
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(
            raw_path.read_text(encoding="utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers=headers,
        )

    # ---- bundle export ----
    @app.get("/bundle.zip")
    def bundle_download():
        okf_root = _store().okf_root
        headers = {"Content-Disposition": 'attachment; filename="distil-bundle.zip"'}
        return StreamingResponse(
            _iter_bundle_zip(okf_root), media_type="application/zip", headers=headers
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
        if not run_delete_entry_stage(entry_id, store):
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
