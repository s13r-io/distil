"""YouTube fetch (Phase 1). ARCHITECTURE.md source-ingest extension; TESTING T-Y*.

Wraps ``yt-dlp`` to (a) enumerate the videos in a playlist and (b) fetch English captions for
one video and parse them into a :class:`~distil.ingest.Transcript` via ``ingest.py``'s existing
SRT parser — so downstream stages (triage/extract/normalize/note) never know the source was
YouTube rather than an uploaded ``.srt``. ``yt-dlp`` is invoked through an injectable ``run``
callable (``subprocess.run`` signature) so callers can fake the process boundary in tests.

Both call sites pin ``--extractor-args youtube:player_client=android,web`` (Phase 19): YouTube
throttles the default ``web`` client hard on datacenter/container egress IPs (429s), and ``web``
also needs yt-dlp to run a JS interpreter to solve its n-parameter signature challenge — which
``python:3.11-slim`` (the Dockerfile base) doesn't have, hence the harmless-but-noisy "No
supported JavaScript runtime" warning. ``android`` needs no JS runtime and isn't throttled nearly
as aggressively, so it goes first; ``web`` stays as a fallback in the same call for the rare
video where android's response lacks the caption track. If a future client chain requires a JS
runtime, that's a real Dockerfile change (install deno), not a default to reach for lightly.

Both call sites also retry a few times with exponential backoff on transient failures (429/5xx/
network) via :func:`_run_yt_dlp` before surfacing :class:`YoutubeFetchError`, to ride out a short
YouTube rate window — bounded so a persistent block still fails fast with the same error.

Speech-to-text (Whisper) for uncaptioned videos is out of scope; those videos raise
:class:`YoutubeFetchError` so callers can skip + report them without failing a whole playlist.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from distil.ingest import IngestError, Transcript, ingest_srt_text
from distil.source import is_youtube_host

_YT_DLP = "yt-dlp"

# android needs no JS runtime and is throttled far less than web on server IPs; web stays as a
# fallback in the same invocation for the rare video whose android response omits captions.
_PLAYER_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=android,web"]

# Bounded retry for transient failures (429 / 5xx / network) — enough attempts to ride out a
# short YouTube rate window, few enough that a persistent block still fails fast.
_TRANSIENT_MAX_ATTEMPTS = 3
_TRANSIENT_BASE_DELAY = 2.0  # seconds; doubles each retry (2s, 4s, ...)

_TRANSIENT_MARKERS = (
    "429",
    "Too Many Requests",
    "500",
    "502",
    "503",
    "504",
    "Internal Server Error",
    "Bad Gateway",
    "Service Unavailable",
    "Gateway Timeout",
    "Connection reset",
    "urlopen error",
    "Read timed out",
    "Temporary failure",
)


class YoutubeFetchError(ValueError):
    """Raised when a playlist can't be listed, or a video has no captions / can't be fetched."""


def _is_transient_failure(stderr: str | None) -> bool:
    text = stderr or ""
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _run_yt_dlp(cmd: list[str], run, timeout: float, sleep=time.sleep):
    """Invoke yt-dlp, retrying transient failures (429/5xx/network) with exponential backoff.

    Returns the last :class:`subprocess.CompletedProcess`-like result whether it succeeded or
    exhausted retries; callers still do their own ``returncode`` check. Non-transient failures
    (e.g. a private/deleted video) return on the first attempt — no point retrying those.
    """
    proc = run(cmd, capture_output=True, text=True, timeout=timeout)
    for attempt in range(_TRANSIENT_MAX_ATTEMPTS - 1):
        if proc.returncode == 0 or not _is_transient_failure(proc.stderr):
            return proc
        sleep(_TRANSIENT_BASE_DELAY * (2**attempt))
        proc = run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc


def is_playlist_url(url: str) -> bool:
    """True for a YouTube playlist link (``?list=...`` with no specific video); false otherwise."""
    if not is_youtube_host(url):
        return False
    candidate = url if "://" in url else f"https://{url}"
    parsed = urlparse(candidate)
    query = parse_qs(parsed.query)
    if "list" in query and "v" not in query:
        return True
    return "/playlist" in parsed.path


def list_playlist_video_urls(
    playlist_url: str, *, run=subprocess.run, timeout: float = 60.0, sleep=time.sleep
) -> list[str]:
    """Enumerate a playlist's videos as normalized ``watch?v=`` URLs (no downloads)."""
    proc = _run_yt_dlp(
        [_YT_DLP, *_PLAYER_CLIENT_ARGS, "--flat-playlist", "--dump-single-json", playlist_url],
        run, timeout, sleep,
    )
    if proc.returncode != 0:
        raise YoutubeFetchError(f"Could not list playlist: {_tail(proc.stderr)}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise YoutubeFetchError("Playlist listing returned invalid data.") from exc
    entries = data.get("entries") or [] if isinstance(data, dict) else []
    urls = [
        f"https://www.youtube.com/watch?v={entry['id']}"
        for entry in entries
        if isinstance(entry, dict) and entry.get("id")
    ]
    if not urls:
        raise YoutubeFetchError("Playlist has no videos.")
    return urls


def fetch_video_transcript(
    video_url: str,
    *,
    run=subprocess.run,
    workdir: str | Path | None = None,
    timeout: float = 120.0,
    sleep=time.sleep,
) -> Transcript:
    """Fetch English captions (prefer ``json3``/native, converted to ``.srt``) for one video."""
    if workdir is not None:
        # A caller-supplied workdir may be reused across fetches (e.g. tests sharing a
        # tmp_path); scope this fetch to its own unique child directory so a stale caption
        # file left behind by a previous invocation is never picked up by the glob below.
        scoped = Path(tempfile.mkdtemp(dir=str(workdir)))
        return _fetch_into(video_url, run, scoped, timeout, sleep)
    with tempfile.TemporaryDirectory() as tmp:
        return _fetch_into(video_url, run, Path(tmp), timeout, sleep)


def _fetch_into(video_url: str, run, workdir: Path, timeout: float, sleep=time.sleep) -> Transcript:
    out_prefix = workdir / "captions"
    proc = _run_yt_dlp(
        [
            _YT_DLP, *_PLAYER_CLIENT_ARGS,
            "--skip-download",
            "--write-subs", "--write-auto-subs",
            # Exact "en" only — a wildcard like "en.*" also matches yt-dlp's auto-translated
            # variants (e.g. "en-de", "en-fr" translated *into* English), which are lower
            # quality than the original/auto-generated English track and would be picked up
            # by the sort-and-take-first below.
            "--sub-langs", "en",
            "--sub-format", "json3/best",
            "--convert-subs", "srt",
            "-o", str(out_prefix),
            video_url,
        ],
        run, timeout, sleep,
    )
    if proc.returncode != 0:
        raise YoutubeFetchError(f"yt-dlp failed: {_tail(proc.stderr)}")
    srt_files = sorted(workdir.glob("*.srt"))
    if not srt_files:
        raise YoutubeFetchError("No English captions available for this video.")
    raw = srt_files[0].read_text(encoding="utf-8")
    try:
        return ingest_srt_text(raw)
    except IngestError as exc:
        raise YoutubeFetchError(str(exc)) from exc


def _tail(text: str | None, limit: int = 300) -> str:
    return (text or "").strip()[:limit] or "unknown error"
