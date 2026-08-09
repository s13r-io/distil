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
import threading
import zipfile
from dataclasses import asdict
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
from distil.cli import (
    _make_canonicalize_client,
    _make_client,
    _make_embedder,
    _make_extract_client,
    _make_graph_client,
    _make_link_client,
    _make_note_client,
    _make_summary_client,
)
from distil.graph import link_graph
from distil.ingest import (
    IngestError,
    Segment,
    Transcript,
    TranscriptTooShortError,
    ingest_file,
    ingest_srt_text,
    ingest_text,
    is_thin_source,
)
from distil.pipeline import PipelineConfig, run_pipeline
from distil.profile_update import apply_feedback
from distil.query import ask as run_ask
from distil.query import stream_ask
from distil.refresh_summary import refresh_narrative_summary
from distil.source import (
    SourceMetadata,
    SourceMetadataError,
    SourceUrlError,
    clean_source_title,
    display_title,
    fetch_youtube_oembed_metadata,
    is_youtube_host,
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
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE = None

_PLAYLIST_FETCH_DELAY_DEFAULT = 3.0  # seconds; see create_app()'s Fetcher wiring for why
_COLLECTOR_LEASE_SECONDS_DEFAULT = jobsmod._DEFAULT_COLLECTOR_LEASE_SECONDS
_COLLECTOR_EXPIRY_SECONDS_DEFAULT = jobsmod._DEFAULT_COLLECTOR_EXPIRY_SECONDS
_COLLECTOR_CLAIM_LIMIT_DEFAULT = 5
_COLLECTOR_CLAIM_LIMIT_MAX = 20


def _db_path() -> str:
    return os.environ.get("DISTIL_DB_PATH", "./data/distil.db")


def _kb_dir() -> str:
    return os.environ.get("DISTIL_KB_DIR", "./kb")


def _store() -> Store:
    return Store(db_path=_db_path(), kb_dir=_kb_dir())


def _staging_root() -> Path:
    """Where in-flight job content (uploads, prefetched transcripts) is staged.

    Derived from ``DISTIL_DB_PATH``'s own directory by default — that's the volume Railway
    (or any host) already mounts persistently (DEPLOY_RAILWAY.md), so staged content survives a
    redeploy without needing a second volume path configured. ``DISTIL_STAGING_DIR`` overrides
    it directly, same convention as ``DISTIL_DB_PATH``/``DISTIL_KB_DIR``.
    """
    override = os.environ.get("DISTIL_STAGING_DIR")
    if override:
        return Path(override)
    return Path(_db_path()).parent / "staging"


def _upload_dir() -> Path:
    d = _staging_root() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transcript_stage_dir() -> Path:
    d = _staging_root() / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _playlist_fetch_delay_seconds() -> float:
    return _env_float("DISTIL_PLAYLIST_FETCH_DELAY_SECONDS", _PLAYLIST_FETCH_DELAY_DEFAULT)


def _collector_lease_seconds() -> float:
    return _env_float("DISTIL_COLLECTOR_LEASE_SECONDS", _COLLECTOR_LEASE_SECONDS_DEFAULT)


def _collector_expiry_seconds() -> float:
    """How long a video waits for external collection before failing cleanly — default 7 days
    (WEB_UI_SPEC collector queue), configurable via ``DISTIL_COLLECTOR_EXPIRY_SECONDS`` the same
    way ``DISTIL_PLAYLIST_FETCH_DELAY_SECONDS`` already is."""
    return _env_float("DISTIL_COLLECTOR_EXPIRY_SECONDS", _COLLECTOR_EXPIRY_SECONDS_DEFAULT)


def _default_profile():
    from distil.models import Profile

    return Profile(user_id=_USER_ID)


def _humanize_tag(tag: str) -> str:
    acronyms = {"ai", "api", "cli", "db", "kb", "llm", "ui", "ux"}
    parts = tag.replace("_", " ").replace("-", " ").split()
    words = [part.upper() if part.lower() in acronyms else part.capitalize() for part in parts]
    return " ".join(words)


_TEMPLATES.env.filters["humanize_tag"] = _humanize_tag

# Human-readable labels for progress display (web/templates/index.html); order here has no
# effect on the declared phase plan, which _build_phase_plan computes per job.
PHASE_LABELS: dict[str, str] = {
    "transcript_fetch": "Fetching transcript",
    "caption_parse": "Parsing captions",
    "ingest": "Reading transcript",
    "metadata": "Fetching video info",
    "extract": "Classifying & extracting knowledge",
    "normalize": "Normalizing items",
    "link": "Linking to profile",
    "note": "Writing teaching note",
    "narrative_summary": "Writing narrative summary",
    "graph": "Linking related entries",
    "file": "Filing entry",
    "canonicalize": "Matching concepts",
    "concept_edges": "Linking concepts",
}


def _build_phase_plan(
    job: jobsmod.Job, *, enable_graph: bool, enable_canonicalize: bool, enable_concept_edges: bool,
    enable_narrative_summary: bool = True,
) -> list[str]:
    """The ordered phases *this* job will actually run, given its kind and the pipeline flags.

    Declaring the total from these flags up front (rather than a fixed 9) is what keeps the
    step count honest for jobs where graph/canonicalize/concept_edges are disabled. Triage is no
    longer its own phase — it's merged into "extract" (one strong-tier call; see pipeline.py's
    module docstring). "narrative_summary" is listed right after "metadata", before "extract",
    reflecting when it actually starts (the cheap-tier summary runs concurrently with "extract",
    not after "note" — its belated "finish" event doesn't disturb this ordering, since
    current_phase only ever advances on a "start" event; see _PhaseReporter.on_phase).
    """
    pre = ["transcript_fetch", "caption_parse"] if job.kind == "youtube" else ["ingest"]
    plan = [*pre, "metadata"]
    if enable_narrative_summary:
        plan.append("narrative_summary")
    plan += ["extract", "normalize", "link", "note"]
    if enable_graph:
        plan.append("graph")
    plan.append("file")
    if enable_canonicalize:
        plan.append("canonicalize")
        if enable_concept_edges:
            plan.append("concept_edges")
    return plan


class _PhaseReporter:
    """Turns stage start/finish events into job-table progress + persisted durations.

    Used both for the pre-pipeline web-layer steps (ingest/transcript_fetch/caption_parse/
    metadata) and, via ``on_pipeline_event``, as ``PipelineConfig.phase_callback`` — one
    reporter instance covers a job's whole run so phase indices stay consistent across both.
    """

    def __init__(self, jobs_store: jobsmod.JobStore, job_id: str, plan: list[str]):
        self._store = jobs_store
        self._job_id = job_id
        self._plan = plan
        self._total = len(plan)
        self._starts: dict[str, float] = {}

    def on_phase(self, phase: str, event: str) -> None:
        if event == "start":
            self._starts[phase] = perf_counter()
            index = self._plan.index(phase) + 1 if phase in self._plan else self._total
            self._store.start_phase(self._job_id, phase=phase, index=index, total=self._total)
        elif event == "finish":
            elapsed = perf_counter() - self._starts.get(phase, perf_counter())
            self._store.record_phase_duration(self._job_id, phase=phase, seconds=elapsed)
        elif event == "short_circuit":
            self._store.collapse_total(self._job_id)

    def on_pipeline_event(self, stage: str, event: str) -> None:
        self.on_phase(stage, event)


def _distill_job(job: jobsmod.Job) -> dict:
    """Worker callback: run the pipeline for one job, return a small result dict.

    Builds fresh Store/client/embedder on the worker thread (no cross-thread sqlite sharing).
    """
    timings: dict[str, float] = {}
    total_start = perf_counter()
    store = _store()
    jobs_store = jobsmod.JobStore(_db_path())
    enable_graph = False
    enable_canonicalize = True
    enable_concept_edges = True
    enable_narrative_summary = True
    plan = _build_phase_plan(
        job, enable_graph=enable_graph, enable_canonicalize=enable_canonicalize,
        enable_concept_edges=enable_concept_edges,
        enable_narrative_summary=enable_narrative_summary,
    )
    reporter = _PhaseReporter(jobs_store, job.job_id, plan)
    profile = store.load_profile(_USER_ID) or _default_profile()
    try:
        transcript = _time_block(
            timings, "ingest", lambda: _load_job_transcript(job, on_phase=reporter.on_phase)
        )
    except TranscriptTooShortError as exc:
        # Not a fetch failure and not a quality judgment call — a plain word-count floor the
        # owner set. Distinguished from a real failure via its own status (STATUS_LOW_VALUE),
        # never STATUS_FAILED. Collapse the declared phase total honestly: nothing past the
        # ingest/fetch step will run. The phase name passed here is never read by
        # collapse_total (it only reads the job's already-persisted phase_index) — it fires
        # equally whether the exception came from local ingest or a YouTube fetch's caption
        # parse.
        reporter.on_phase("ingest", "short_circuit")
        return {"status": jobsmod.STATUS_LOW_VALUE, "entry_id": None, "summary": str(exc)}
    except YoutubeFetchError as exc:
        # Only a bot-check refusal is worth waiting on — a genuinely uncaptioned/private/missing
        # video must still fail immediately, exactly as it does today (see
        # distil.youtube.is_bot_check_refusal for how confidently these are told apart).
        if youtube.is_bot_check_refusal(exc):
            return {"status": "awaiting_collection", "error": str(exc)}
        raise
    client = _make_client()
    embedder = _time_block(timings, "embedder", _cached_safe_embedder)
    reporter.on_phase("metadata", "start")
    source_meta = _time_block(timings, "metadata", lambda: _fetch_source_metadata(job.source_url))
    reporter.on_phase("metadata", "finish")
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
            enable_graph=enable_graph,
            enable_canonicalize=enable_canonicalize,
            enable_concept_edges=enable_concept_edges,
            enable_narrative_summary=enable_narrative_summary,
            timing_callback=lambda stage, seconds: timings.__setitem__(stage, seconds),
            phase_callback=reporter.on_pipeline_event,
        ),
        embedder=embedder,
        summary_client=_make_summary_client(),
        extract_client=_make_extract_client(),
        link_client=_make_link_client(),
        note_client=_make_note_client(),
        graph_client=_make_graph_client(),
        canonicalize_client=_make_canonicalize_client(),
    )
    total = perf_counter() - total_start
    n = len(entry.knowledge_items)
    # No quality short-circuit: run_pipeline always files an entry once it starts (Stage 7 runs
    # unconditionally) — this is always STATUS_DONE, even when extraction genuinely found nothing.
    graph_scheduled = _schedule_graph_link(entry.entry_id) if entry.tags.topics else False
    graph_note = " · graph updating" if graph_scheduled else ""
    _emit_timing_log(job, entry.entry_id, jobsmod.STATUS_DONE, entry.triage.verdict, n,
                     timings, total)
    return {"status": jobsmod.STATUS_DONE, "entry_id": entry.entry_id,
            "summary": f"kept {n} item{'s' if n != 1 else ''} · verdict {entry.triage.verdict} "
                       f"· {_format_timings(timings, total)}{graph_note}"}


def _load_job_transcript(job: jobsmod.Job, *, on_phase=None):
    if job.kind == "youtube":
        # youtube.fetch_video_transcript reports its own transcript_fetch/caption_parse
        # start/finish pair around the yt-dlp call and the srt parse respectively. Only a
        # single-video submission still reaches here — a playlist video's transcript is fetched
        # up front by the Fetcher (see _fetch_playlist_video) and arrives as kind
        # KIND_YOUTUBE_STAGED below, so this branch never blocks the rest of a playlist.
        return youtube.fetch_video_transcript(job.payload, on_phase=on_phase)
    if on_phase is not None:
        on_phase("ingest", "start")
    if job.kind == "file":
        p = Path(job.payload)
        if not p.exists():
            raise FileNotFoundError("Source content no longer available — please re-add.")
        result = ingest_file(str(p))
    elif job.kind == jobsmod.KIND_YOUTUBE_STAGED:
        p = Path(job.payload)
        if not p.exists():
            raise FileNotFoundError("Source content no longer available — please re-add.")
        result = _load_staged_transcript(p)
    else:
        result = ingest_text(job.payload)
    if on_phase is not None:
        on_phase("ingest", "finish")
    return result


def _fetch_playlist_video(job: jobsmod.Job) -> dict:
    """Fetcher worker callback: fetch one playlist video's transcript and stage it to the
    persistent volume. Never raises — a fetch failure becomes a ``failed`` result so it fails
    only this job, exactly like the old inline-fetch-at-distill-time path did (WEB_UI_SPEC §8 /
    _enqueue_youtube_source docstring) — except specifically a YouTube bot-check refusal, which
    becomes ``awaiting_collection`` instead: parked for an external collector rather than failed,
    since that failure is about this server's address, not this video; and a transcript below the
    owner's word-count floor, which becomes ``low_value`` — a clean rejection, not a failure."""
    jobs_store = jobsmod.JobStore(_db_path())
    reporter = _PhaseReporter(jobs_store, job.job_id, ["transcript_fetch", "caption_parse"])
    try:
        transcript = youtube.fetch_video_transcript(job.payload, on_phase=reporter.on_phase)
    except TranscriptTooShortError as exc:
        reporter.on_phase("caption_parse", "short_circuit")
        return {"status": jobsmod.STATUS_LOW_VALUE, "summary": str(exc)}
    except YoutubeFetchError as exc:
        if youtube.is_bot_check_refusal(exc):
            return {"status": "awaiting_collection", "error": str(exc)}
        return {"status": "failed", "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Fetching the transcript timed out."}
    except OSError as exc:
        return {"status": "failed", "error": f"Could not run yt-dlp: {exc}"}
    staged_path = _stage_transcript(job.job_id, transcript)
    return {"status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED, "payload": str(staged_path)}


def _stage_transcript(job_id: str, transcript: Transcript) -> Path:
    """Persist a fetched Transcript to the volume as JSON — the exact shape ingest.py already
    produces, so reading it back needs no re-parsing, just reconstruction (_load_staged_transcript)."""
    path = _transcript_stage_dir() / f"{job_id}.json"
    data = {"segments": [asdict(s) for s in transcript.segments]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _load_staged_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(segments=[Segment(**s) for s in data["segments"]])


def _staged_path_for_job(job: jobsmod.Job) -> Path | None:
    """The on-disk file (if any) a job's payload points at — the two kinds staged on the
    persistent volume rather than held inline. ``missing_ok`` unlinking elsewhere makes it safe
    to call this even after the file's already been consumed."""
    if job.kind in ("file", jobsmod.KIND_YOUTUBE_STAGED):
        return Path(job.payload)
    return None


def _cleanup_staged_file(job: jobsmod.Job) -> None:
    path = _staged_path_for_job(job)
    if path is not None:
        path.unlink(missing_ok=True)


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
        "ingest", "metadata", "narrative_summary", "extract", "normalize", "link", "note",
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
        related = link_graph(entry, store, _make_graph_client())
        if related:
            entry.related_entries = related
            store.file_entry(entry)
    except Exception:
        return


def _enqueue_youtube_source(store_jobs: jobsmod.JobStore, url: str):
    """ADD input: a bare YouTube video or playlist URL, no paste/file.

    A playlist enqueues one ``youtube`` job per video, each starting in
    ``STATUS_PENDING_FETCH`` rather than ``queued``: the ``Fetcher`` background thread (jobs.py)
    fetches every video's transcript up front, one at a time with a pause between (Phase E —
    see ``_fetch_playlist_video``), staging each onto the persistent volume as it succeeds and
    flipping that job to ``queued`` (kind ``youtube_staged``) for the still-single-worker distill
    ``Worker`` to pick up. Distilling of the first video therefore starts as soon as it's fetched,
    while the rest of the playlist keeps fetching in the background (the two workers claim
    disjoint job statuses, so neither blocks the other). A bad video's caption/availability
    failure only fails its own job (``Fetcher``/``Worker`` both isolate per-job exceptions), so
    one skipped video never blocks the rest of the playlist.
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
                status=jobsmod.STATUS_PENDING_FETCH,
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


def _entity_detail_context(store: Store, entity) -> dict:
    """Template context for an entity detail page — the exact ``_concept_detail_context``
    shape, one granularity down (Phase D), minus typed edges (entities have none)."""
    member_entries: dict[str, object] = {}
    for member in entity.members:
        if member.entry_id in member_entries:
            continue
        try:
            member_entries[member.entry_id] = store.load_entry(member.entry_id)
        except Exception:
            continue

    members_by_item = {m.item_id: m for m in entity.members}
    claims = []
    for claim in entity.claims:
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
        claims.append({"text": claim.text, "citations": citations})

    sources = []
    seen_entries: set[str] = set()
    for member in entity.members:
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

    return {"entity": entity, "claims": claims, "sources": sources}


def _entities_for_entry(store: Store, entry_id: str) -> list[dict]:
    return [
        {"entity_id": e.entity_id, "title": e.title}
        for e in store.list_entities()
        if any(m.entry_id == entry_id for m in e.members)
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
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # on_finished only fires for a successful (done/low_value) terminal status — a failed job's
    # staged file must stay on disk so a retry can still read it (see _load_job_transcript).
    worker = jobsmod.Worker(
        _db_path(), _distill_job, on_finished=_cleanup_staged_file,
        collector_expiry_seconds=_collector_expiry_seconds(),
    )
    # A second, independent single-worker thread (Phase E): fetches playlist videos'
    # transcripts up front, overlapping with `worker` distilling whatever's already staged —
    # they claim disjoint job statuses (pending_fetch/fetching vs queued/running), so neither
    # blocks the other, and distilling stays exactly as single-threaded as before.
    fetcher = jobsmod.Fetcher(
        _db_path(), _fetch_playlist_video, delay_seconds=_playlist_fetch_delay_seconds(),
        collector_expiry_seconds=_collector_expiry_seconds(),
    )

    @app.on_event("startup")
    def _start_worker():
        worker.start()
        fetcher.start()

    @app.on_event("shutdown")
    def _stop_worker():
        worker.stop()
        fetcher.stop()

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        # /collector/* routes are exempt from the owner auth gate: they're gated by their own,
        # separate credential (auth.request_is_collector_authorized), checked inside each route —
        # never by the owner's session cookie or DISTIL_AUTH_SECRET bearer token.
        if (
            not auth.path_is_open(request.url.path)
            and not request.url.path.startswith("/static")
            and not auth.path_is_collector(request.url.path)
        ):
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

    def _entities_template_context() -> dict:
        entities = [
            {
                "entity_id": e.entity_id,
                "kind": e.kind,
                "title": e.title,
                "description": e.description,
                "video_count": len({m.entry_id for m in e.members}),
            }
            for e in _store().list_entities()
        ]
        return {"entities": entities, "entity_count": len(entities)}

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

    # ---- entities ----
    @app.get("/entities", response_class=HTMLResponse)
    def entities_list(request: Request):
        return _TEMPLATES.TemplateResponse(
            request, "entities.html",
            {**_entities_template_context(), "active_page": "entities"},
        )

    @app.get("/entities/{entity_id}", response_class=HTMLResponse)
    def entity_page(request: Request, entity_id: str):
        store = _store()
        entity = store.load_entity(entity_id)
        if entity is None:
            return HTMLResponse("<p>Entity not found.</p>", status_code=404)
        return _TEMPLATES.TemplateResponse(
            request, "entity.html",
            {**_entity_detail_context(store, entity), "active_page": "entities"},
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
            dest = _upload_dir() / f"{os.urandom(6).hex()}{suffix}"
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
        store_jobs = jobsmod.JobStore(_db_path())
        # A failed job's staged file is kept for retry(), but must not accumulate on the volume
        # forever if nobody retries or removes it — reap it once the job has sat failed past the
        # same 24h bound list_active()'s autoclear already uses for finished rows.
        store_jobs.autoclear(on_stale_failed=_cleanup_staged_file)
        # A global signal, not a per-job one — attached to every row so the Activity view can
        # render an honest "is anything actually coming for this?" for a parked video without a
        # second round trip or changing /jobs from a bare array into an object.
        last_collector_checkin = store_jobs.last_collector_checkin()
        return [
            {
                **j.to_dict(),
                "collector_last_seen": last_collector_checkin,
                "presentation": jobsmod.status_presentation(j.status),
                "collector_status": (
                    jobsmod.collector_status_for_job(j, last_collector_checkin)
                    if j.status in (jobsmod.STATUS_AWAITING_COLLECTION, jobsmod.STATUS_COLLECTING)
                    else None
                ),
            }
            for j in store_jobs.list_active()
        ]

    @app.post("/jobs/{job_id}/remove")
    def jobs_remove(job_id: str):
        store_jobs = jobsmod.JobStore(_db_path())
        job = store_jobs.get(job_id)
        ok = store_jobs.remove_queued(job_id)
        # A removed job never ran, so nothing already cleaned up its staged file (uploaded file
        # or prefetched transcript) — do it now rather than leaking it on the persistent volume.
        if ok and job is not None:
            _cleanup_staged_file(job)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

    @app.post("/jobs/{job_id}/retry")
    def jobs_retry(job_id: str):
        ok = jobsmod.JobStore(_db_path()).retry(job_id)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

    @app.post("/jobs/clear")
    def jobs_clear(scope: str = "finished"):
        store_jobs = jobsmod.JobStore(_db_path())
        statuses = (
            {jobsmod.STATUS_DONE, jobsmod.STATUS_LOW_VALUE, jobsmod.STATUS_REMOVED}
            if scope == "finished"
            else {jobsmod.STATUS_FAILED} if scope == "failed" else set()
        )
        # A done/low_value job's staged file was already unlinked on success, and a removed one
        # at removal time — but a failed job's file is deliberately kept for retry(), so clearing
        # failed rows here is the one place that still needs to clean it up.
        for job in store_jobs.list_active():
            if job.status in statuses:
                _cleanup_staged_file(job)
        n = store_jobs.clear(scope)
        return {"cleared": n}

    # ---- external collector queue (bot-check refusals only) ----
    # Gated by auth.request_is_collector_authorized, a separate scoped credential
    # (DISTIL_COLLECTOR_TOKEN) — never the owner's session/bearer (see _auth_gate above, which
    # exempts this whole prefix from the owner check specifically so these routes can enforce
    # their own). A collector can claim and submit here and reach nothing else on the site.
    def _require_collector_auth(request: Request) -> JSONResponse | None:
        if not auth.request_is_collector_authorized(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return None

    @app.post("/collector/jobs/claim")
    def collector_claim(request: Request, limit: int = _COLLECTOR_CLAIM_LIMIT_DEFAULT):
        unauthorized = _require_collector_auth(request)
        if unauthorized is not None:
            return unauthorized
        bounded_limit = max(1, min(limit, _COLLECTOR_CLAIM_LIMIT_MAX))
        store_jobs = jobsmod.JobStore(_db_path())
        # Recorded even when nothing is claimed below — an empty claim still proves a collector
        # process is alive and polling, which is the one signal the owner's Activity view needs.
        store_jobs.record_collector_checkin()
        claimed = store_jobs.claim_for_collection(
            limit=bounded_limit, lease_seconds=_collector_lease_seconds(),
        )
        return {
            "jobs": [
                {"job_id": j.job_id, "url": j.payload, "lease_expires_at": j.lease_expires_at}
                for j in claimed
            ]
        }

    @app.post("/collector/jobs/{job_id}/transcript")
    def collector_submit_transcript(request: Request, job_id: str, srt: str = Form(...)):
        unauthorized = _require_collector_auth(request)
        if unauthorized is not None:
            return unauthorized
        store_jobs = jobsmod.JobStore(_db_path())
        # Validated as parseable captions before anything is accepted into the pipeline — the
        # same parser a real yt-dlp fetch's captions already go through (distil.youtube._fetch_into).
        try:
            transcript = ingest_srt_text(srt)
        except TranscriptTooShortError as exc:
            # A clean rejection, not a parse failure — resolve the job now (STATUS_LOW_VALUE)
            # instead of just 400ing and leaving it to be re-claimed and re-submitted in a loop
            # until its 7-day collection_deadline gives up. Idempotent: a retried submit of the
            # same too-short transcript finds the job already resolved and skips the update.
            job = store_jobs.get(job_id)
            if job is not None and job.status == jobsmod.STATUS_COLLECTING:
                store_jobs.mark_low_value(job_id, entry_id=None, summary=str(exc))
            return {"job_id": job_id, "status": jobsmod.STATUS_LOW_VALUE, "detail": str(exc)}
        except IngestError as exc:
            return JSONResponse(
                {"detail": f"Not parseable as captions: {exc}"}, status_code=400
            )
        # Validate the job against the DB before ever writing to a path built from the raw,
        # collector-controlled job_id — mirrors jobs_remove's get-then-touch-file ordering rather
        # than staging first and cleaning up after the fact.
        job = store_jobs.get(job_id)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        already_submitted = job.status == jobsmod.STATUS_QUEUED or job.collected_at is not None
        if already_submitted:
            # Nothing to (re-)write: the transcript is already in the pipeline (or the job has
            # since finished/failed and any staged file was already cleaned up) — staging again
            # here would recreate a file that no future event will ever clean up again.
            return {"job_id": job_id, "status": jobsmod.STATUS_QUEUED}
        if job.status != jobsmod.STATUS_COLLECTING:
            return JSONResponse(
                {"detail": "job is not currently leased to a collector"}, status_code=409
            )
        staged_path = _stage_transcript(job_id, transcript)
        outcome = store_jobs.submit_collected_transcript(job_id, staged_path=str(staged_path))
        if outcome in ("not_found", "not_leased"):
            # The DB state moved between the check above and the guarded update (e.g. the lease
            # expired mid-request) — nothing owns this file, so remove it rather than leak it on
            # the volume forever under this job_id's deterministic path.
            staged_path.unlink(missing_ok=True)
            status_code = 404 if outcome == "not_found" else 409
            detail = "job not found" if outcome == "not_found" else (
                "job is not currently leased to a collector"
            )
            return JSONResponse({"detail": detail}, status_code=status_code)
        # "accepted" or "already_submitted" (idempotent retry) both report success — the staged
        # path is deterministic per job_id, so a duplicate submit just overwrites it in place.
        return {"job_id": job_id, "status": jobsmod.STATUS_QUEUED}

    @app.post("/collector/jobs/{job_id}/unfetchable")
    def collector_report_unfetchable(request: Request, job_id: str, reason: str = Form(...)):
        unauthorized = _require_collector_auth(request)
        if unauthorized is not None:
            return unauthorized
        store_jobs = jobsmod.JobStore(_db_path())
        ok = store_jobs.report_uncollectable(job_id, error=reason.strip()[:500])
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

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
        thin_source = is_thin_source(e.source.transcript_word_count)
        slug = okf.slug_for_entry(e, store.okf_root)
        has_transcript = (store.okf_root / "raw" / f"{slug}.md").exists()
        job = jobsmod.JobStore(_db_path()).find_by_entry_id(entry_id)
        phase_durations = job.phase_durations if job else {}
        stage_timings = [
            (PHASE_LABELS.get(stage, stage), seconds)
            for stage, seconds in sorted(phase_durations.items(), key=lambda kv: -kv[1])
        ]
        return _TEMPLATES.TemplateResponse(
            request, "entry.html",
            {"e": e, "mix": mix,
             "thin_source": thin_source,
             "has_transcript": has_transcript,
             "concepts_for_entry": _concepts_for_entry(store, entry_id),
             "entities_for_entry": _entities_for_entry(store, entry_id),
             "reasons": ["relevant", "already_knew", "bad_source", "wrong_for_me",
                         "irrelevant_now"],
             "active_page": "library",
             "stage_timings": stage_timings,
             "total_processing_seconds": round(sum(phase_durations.values()), 1),
             },
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

    @app.post("/entries/{entry_id}/refresh-summary")
    def refresh_summary(entry_id: str):
        """Generate/regenerate ONLY this entry's narrative summary, from its stored raw
        transcript — never re-fetches from YouTube, never re-runs extraction (see
        distil/refresh_summary.py). Reports plainly (200 with ok=false) when no stored
        transcript is available, rather than a generic error."""
        store = _store()
        result = refresh_narrative_summary(entry_id, store, _make_summary_client())
        if not result.ok and result.message.startswith("Entry"):
            return JSONResponse({"detail": result.message}, status_code=404)
        return JSONResponse({"ok": result.ok, "message": result.message})

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

    # ---- diagnostics (Phase 23/24) ----
    @app.get("/diagnostics/youtube-pot")
    def diagnose_youtube_pot(url: str):
        """Run a verbose yt-dlp fetch for ``url`` and report PO-token provider discovery,
        per-context attempts, and whether the bot-check safety net recognizes this run's output
        — the permanent replacement for SSH'ing in to run this by hand (see
        distil/youtube.py's module docstring, Phase 23/24)."""
        if not is_youtube_host(url):
            return JSONResponse({"detail": "url must be a YouTube URL."}, status_code=400)
        result = youtube.diagnose_pot(url)
        return JSONResponse({
            "returncode": result.returncode,
            "provider_discovery": result.provider_discovery,
            "context_attempts": [
                {"context": context, "client": client}
                for context, client in result.context_attempts
            ],
            "bot_check_detected": result.bot_check_detected,
            "raw_output": result.raw_output,
        })

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
        "concepts": [
            {"concept_id": c.concept_id, "title": c.title} for c in result.concepts
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
