"""YouTube fetch (Phase 1). ARCHITECTURE.md source-ingest extension; TESTING T-Y*.

Wraps ``yt-dlp`` to (a) enumerate the videos in a playlist and (b) fetch English captions for
one video and parse them into a :class:`~distil.ingest.Transcript` via ``ingest.py``'s existing
SRT parser — so downstream stages (triage/extract/normalize/note) never know the source was
YouTube rather than an uploaded ``.srt``. ``yt-dlp`` is invoked through an injectable ``run``
callable (``subprocess.run`` signature) so callers can fake the process boundary in tests.

Both call sites pin ``--extractor-args youtube:player_client=android_vr,web_safari`` (Phase 21;
Phase 19 originally pinned ``android,web``). ``android`` was dropped from yt-dlp's own default
client list years ago and is what produces the "SABR-only streaming experiment" warning on
today's yt-dlp; ``android_vr`` is its currently-recommended, still-unauthenticated replacement —
it's yt-dlp's own ``_DEFAULT_JSLESS_CLIENTS`` (see ``yt_dlp.extractor.youtube._video``), i.e. what
yt-dlp itself picks when no JS runtime is available, which is exactly our situation:
``python:3.11-slim`` (the Dockerfile base) has none, and android_vr's ``REQUIRE_JS_PLAYER`` is
``False``, needing no interpreter to solve the n-parameter signature challenge that the plain
``web`` client would. ``web_safari`` — yt-dlp's other default-chain member (its full
``_DEFAULT_CLIENTS``) — stays as a fallback in the same call for the rare video whose android_vr
response lacks the caption track; captions don't need the JS-gated format-signature step, so
pairing it with a JS-requiring fallback is safe even with no JS runtime installed. If a future
client chain genuinely requires a JS runtime for *captions* (not formats), that's a real
Dockerfile change (install deno), not a default to reach for lightly. Re-derive this chain from
``yt_dlp.extractor.youtube._video``'s ``_DEFAULT_CLIENTS``/``_DEFAULT_JSLESS_CLIENTS`` next time
it needs revisiting — yt-dlp changes its recommended clients every few months as YouTube reacts.

Both call sites request captions as ``srt`` directly (``--sub-format srt/best``, Phase 21) rather
than a format needing ``--convert-subs``: YouTube's timedtext endpoint serves ``srt`` natively for
both manual and auto-generated tracks, and the only conversions yt-dlp can do *without* shelling
out to ffmpeg are ttml/dfxp — ffmpeg itself isn't installed in the Dockerfile image, so any format
requiring it (the previous ``json3`` choice included) fails outright in production once actually
exercised. Requesting ``srt`` natively sidesteps needing ffmpeg at all.

Both call sites also retry a few times with exponential backoff on transient failures (429/5xx/
network) via :func:`_run_yt_dlp` before surfacing :class:`YoutubeFetchError`, to ride out a short
YouTube rate window — bounded so a persistent block still fails fast with the same error.

Both call sites pass ``--no-update`` to suppress yt-dlp's own "is more than 90 days old" staleness
warning, and surface failures via :func:`_surface_error` (Phase 21), which prefers yt-dlp's
``ERROR:``-prefixed line(s) over anything else in stderr — warnings (SABR, staleness,
impersonation, ...) must never crowd out the real failure the way the previous head-truncating
``_tail`` helper let them. The complete, untruncated stderr is always logged via the standard
``logging`` module (this repo has no other logging convention yet) so a production failure never
again needs a code change just to see what actually happened; only the short user-facing message
stays bounded.

Phase 22: client rotation alone can't beat YouTube's bot check on a *datacenter* IP ("Sign in to
confirm you're not a bot") — that's an identity/reputation challenge, not a throttle, and nothing
in ``--extractor-args youtube:...`` can satisfy it (a prior attempt, ``DISTIL_YOUTUBE_API_KEY``,
only ever substituted into InnerTube's anonymous ``key=`` query parameter and carried no identity;
it has been removed). A proof-of-origin (PO) token from a real attestation provider is yt-dlp's
supported answer: when ``DISTIL_POT_PROVIDER_URL`` is set, :func:`_extractor_args` (shared by both
call sites, read at call time so it can't go stale mid-process) appends a *second*
``--extractor-args`` pair — ``youtubepot-bgutilhttp:base_url=<url>`` — pointing yt-dlp's
``bgutil-ytdlp-pot-provider`` plugin (installed via the ``youtube`` extra) at a separately-run
``bgutil-ytdlp-pot-provider`` HTTP server (a second service; see DEPLOY_RAILWAY.md). This is its
own extractor namespace, not a suffix on the ``youtube:`` value, per yt-dlp's own multi-use
support for ``--extractor-args``. Absent the env var, the command line is byte-identical to
before. Per upstream's own documentation, a PO token makes traffic look more legitimate — it does
not *guarantee* clearing a bot check, so this is unproven against the live datacenter IP until
deployed; it is not a fix that can be verified from a residential dev machine.

Speech-to-text (Whisper) for uncaptioned videos is out of scope; those videos raise
:class:`YoutubeFetchError` so callers can skip + report them without failing a whole playlist.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from distil.ingest import IngestError, Transcript, ingest_srt_text
from distil.source import is_youtube_host

_logger = logging.getLogger(__name__)

_YT_DLP = "yt-dlp"

# android_vr needs no JS runtime (yt-dlp's own no-JS default; see module docstring) and is
# throttled far less than web on server IPs; web_safari stays as a fallback in the same
# invocation for the rare video whose android_vr response omits captions.
_PLAYER_CLIENT = "player_client=android_vr,web_safari"

# The bgutil-ytdlp-pot-provider plugin's own extractor-args namespace (separate from `youtube:`).
_POT_PROVIDER_EXTRACTOR_KEY = "youtubepot-bgutilhttp"


def _extractor_args() -> list[str]:
    """Build the ``--extractor-args`` pairs shared by both yt-dlp call sites.

    Reads ``DISTIL_POT_PROVIDER_URL`` at call time (not import time) so callers — including
    tests via monkeypatch — always see the current environment. When unset, the returned args
    are byte-identical to before (``player_client`` only). When set, appends a second
    ``--extractor-args`` pair pointing the bgutil POT-provider plugin at that server's
    ``base_url`` (yt-dlp supports repeating ``--extractor-args`` for different extractors).
    """
    args = ["--extractor-args", f"youtube:{_PLAYER_CLIENT}"]
    provider_url = os.environ.get("DISTIL_POT_PROVIDER_URL")
    if provider_url:
        args += ["--extractor-args", f"{_POT_PROVIDER_EXTRACTOR_KEY}:base_url={provider_url}"]
    return args


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
        [
            _YT_DLP,
            "--no-update",
            *_extractor_args(),
            "--flat-playlist",
            "--dump-single-json",
            playlist_url,
        ],
        run,
        timeout,
        sleep,
    )
    if proc.returncode != 0:
        _logger.error("yt-dlp playlist listing failed for %s:\n%s", playlist_url, proc.stderr)
        raise YoutubeFetchError(f"Could not list playlist: {_surface_error(proc.stderr)}")
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
    """Fetch English captions (native ``srt``, no ffmpeg conversion needed) for one video."""
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
            _YT_DLP,
            "--no-update",
            *_extractor_args(),
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            # Exact "en" only — a wildcard like "en.*" also matches yt-dlp's auto-translated
            # variants (e.g. "en-de", "en-fr" translated *into* English), which are lower
            # quality than the original/auto-generated English track and would be picked up
            # by the sort-and-take-first below.
            "--sub-langs",
            "en",
            # srt natively — never json3/vtt/etc with --convert-subs: ffmpeg isn't installed
            # in the Dockerfile image, and ffmpeg can't parse yt-dlp's json3 format at all
            # (only ttml/dfxp convert without it). YouTube's timedtext endpoint serves srt
            # directly for both manual and auto-generated tracks, so no conversion is needed.
            "--sub-format",
            "srt/best",
            "-o",
            str(out_prefix),
            video_url,
        ],
        run,
        timeout,
        sleep,
    )
    if proc.returncode != 0:
        _logger.error("yt-dlp fetch failed for %s:\n%s", video_url, proc.stderr)
        raise YoutubeFetchError(f"yt-dlp failed: {_surface_error(proc.stderr)}")
    srt_files = sorted(workdir.glob("*.srt"))
    if not srt_files:
        raise YoutubeFetchError("No English captions available for this video.")
    raw = srt_files[0].read_text(encoding="utf-8")
    try:
        return ingest_srt_text(raw)
    except IngestError as exc:
        raise YoutubeFetchError(str(exc)) from exc


def _surface_error(stderr: str | None, limit: int = 300) -> str:
    """Pull the actionable failure out of yt-dlp stderr for the bounded user-facing message.

    Prefers yt-dlp's own ``ERROR:``-prefixed line(s) — warnings (SABR, staleness, impersonation,
    ...) must never crowd them out. Falls back to a genuine tail (the *last* ``limit`` chars, not
    the first) when yt-dlp didn't emit an ``ERROR:`` line at all, e.g. a bare traceback. Callers
    are expected to log the full ``stderr`` separately before calling this — this function only
    ever returns a bounded string.
    """
    text = (stderr or "").strip()
    if not text:
        return "unknown error"
    error_lines = [line.strip() for line in text.splitlines() if line.lstrip().startswith("ERROR:")]
    if error_lines:
        return "\n".join(error_lines)[:limit]
    return text[-limit:]
