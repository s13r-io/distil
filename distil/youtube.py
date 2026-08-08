"""YouTube fetch (Phase 1). ARCHITECTURE.md source-ingest extension; TESTING T-Y*.

Wraps ``yt-dlp`` to (a) enumerate the videos in a playlist and (b) fetch English captions for
one video and parse them into a :class:`~distil.ingest.Transcript` via ``ingest.py``'s existing
SRT parser — so downstream stages (triage/extract/normalize/note) never know the source was
YouTube rather than an uploaded ``.srt``. ``yt-dlp`` is invoked through an injectable ``run``
callable (``subprocess.run`` signature) so callers can fake the process boundary in tests.

Speech-to-text (Whisper) for uncaptioned videos is out of scope; those videos raise
:class:`YoutubeFetchError` so callers can skip + report them without failing a whole playlist.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from distil.ingest import IngestError, Transcript, ingest_srt_text

_YT_DLP = "yt-dlp"


class YoutubeFetchError(ValueError):
    """Raised when a playlist can't be listed, or a video has no captions / can't be fetched."""


def is_playlist_url(url: str) -> bool:
    """True for a playlist link (``?list=...`` with no specific video); false for a single video."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "list" in query and "v" not in query:
        return True
    return "/playlist" in parsed.path


def list_playlist_video_urls(
    playlist_url: str, *, run=subprocess.run, timeout: float = 60.0
) -> list[str]:
    """Enumerate a playlist's videos as normalized ``watch?v=`` URLs (no downloads)."""
    proc = run(
        [_YT_DLP, "--flat-playlist", "--dump-single-json", playlist_url],
        capture_output=True, text=True, timeout=timeout,
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
) -> Transcript:
    """Fetch English captions (prefer ``json3``/native, converted to ``.srt``) for one video."""
    if workdir is not None:
        return _fetch_into(video_url, run, Path(workdir), timeout)
    with tempfile.TemporaryDirectory() as tmp:
        return _fetch_into(video_url, run, Path(tmp), timeout)


def _fetch_into(video_url: str, run, workdir: Path, timeout: float) -> Transcript:
    out_prefix = workdir / "captions"
    proc = run(
        [
            _YT_DLP, "--skip-download",
            "--write-subs", "--write-auto-subs",
            # Exact "en" only — a wildcard like "en.*" also matches yt-dlp's auto-translated
            # variants (e.g. "en-de", "en-fr" translated *into* English), which are lower
            # quality than the original/auto-generated English track and would be picked up
            # by the sort-and-take-first below.
            "--sub-langs", "en",
            "--sub-format", "srt/best",
            "--convert-subs", "srt",
            "-o", str(out_prefix),
            video_url,
        ],
        capture_output=True, text=True, timeout=timeout,
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
