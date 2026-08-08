"""Phase 1 — youtube.py: fetch captions/enumerate playlists via yt-dlp. Tests T-Y1..Y6.

``yt-dlp`` is invoked through an injected ``run`` callable (``subprocess.run`` signature) so
these stay unit tests: no network, no subprocess, no real binary required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distil.ingest import Transcript
from distil.youtube import (
    YoutubeFetchError,
    fetch_video_transcript,
    is_playlist_url,
    list_playlist_video_urls,
)


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---- T-Y1: playlist URL detection ----


@pytest.mark.unit
def test_is_playlist_url_detects_list_param_without_video():
    assert is_playlist_url("https://www.youtube.com/playlist?list=PL123") is True


@pytest.mark.unit
def test_is_playlist_url_false_for_single_video_with_list_param():
    # A "watch?v=...&list=..." link is a video inside a playlist context, not a playlist URL.
    assert is_playlist_url("https://www.youtube.com/watch?v=abc&list=PL123") is False


@pytest.mark.unit
def test_is_playlist_url_false_for_plain_video():
    assert is_playlist_url("https://youtu.be/abc") is False


# ---- T-Y2: playlist enumeration ----


@pytest.mark.unit
def test_list_playlist_video_urls_returns_watch_urls():
    payload = json.dumps({"entries": [{"id": "abc"}, {"id": "def"}]})

    def fake_run(cmd, **kwargs):
        assert "--flat-playlist" in cmd
        return _proc(returncode=0, stdout=payload)

    urls = list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)
    assert urls == [
        "https://www.youtube.com/watch?v=abc",
        "https://www.youtube.com/watch?v=def",
    ]


@pytest.mark.unit
def test_list_playlist_video_urls_passes_player_client_fallback_chain(monkeypatch):
    monkeypatch.delenv("DISTIL_YOUTUBE_API_KEY", raising=False)
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android,web"
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_list_playlist_video_urls_raises_on_yt_dlp_failure():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr="playlist does not exist")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls(
            "https://www.youtube.com/playlist?list=bad", run=fake_run, sleep=lambda s: None
        )


@pytest.mark.unit
def test_list_playlist_video_urls_raises_when_empty():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0, stdout=json.dumps({"entries": []}))

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=empty", run=fake_run)


# ---- T-Y3: single video transcript fetch (happy path) ----


@pytest.mark.unit
def test_fetch_video_transcript_parses_downloaded_srt(tmp_path):
    srt_body = (
        "1\n00:00:01,000 --> 00:00:03,000\nWelcome to the talk.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nLet's get started.\n"
    )

    def fake_run(cmd, **kwargs):
        # yt-dlp writes the caption file next to the -o output template.
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    transcript = fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path
    )
    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Welcome to the talk."
    assert transcript.segments[0].timestamp == "00:00:01"


@pytest.mark.unit
def test_fetch_video_transcript_passes_player_client_fallback_chain(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTIL_YOUTUBE_API_KEY", raising=False)
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHello.\n"

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android,web"
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


# ---- T-Y7: a reused workdir's stale caption file is never mistaken for current output ----


@pytest.mark.unit
def test_fetch_video_transcript_ignores_stale_srt_in_reused_workdir(tmp_path):
    # Simulate a leftover caption file from a previous fetch that reused this same workdir.
    stale = tmp_path / "captions.en.srt"
    stale.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nSTALE OLD CAPTION\n", encoding="utf-8"
    )

    fresh_srt = "1\n00:00:01,000 --> 00:00:03,000\nFresh caption for this fetch.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(fresh_srt, encoding="utf-8")
        return _proc(returncode=0)

    transcript = fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path
    )
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Fresh caption for this fetch."
    assert "STALE" not in transcript.full_text()
    # The stale file at the workdir root is untouched — the fetch was scoped to a subdirectory.
    assert stale.read_text(encoding="utf-8").count("STALE") == 1


# ---- T-Y4: no captions available -> clear, catchable error ----


@pytest.mark.unit
def test_fetch_video_transcript_raises_when_no_captions_written(tmp_path):
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0)  # yt-dlp "succeeds" but writes no subtitle file

    with pytest.raises(YoutubeFetchError, match="[Cc]aptions"):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=nocaps", run=fake_run, workdir=tmp_path
        )


# ---- T-Y5: yt-dlp process failure (private/deleted video) -> clear error ----


@pytest.mark.unit
def test_fetch_video_transcript_raises_on_yt_dlp_failure(tmp_path):
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr="Video unavailable")

    with pytest.raises(YoutubeFetchError, match="Video unavailable"):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=gone", run=fake_run, workdir=tmp_path
        )


# ---- Phase 19: retry/backoff on transient failures (429/5xx/network) ----


@pytest.mark.unit
def test_fetch_video_transcript_retries_transient_429_then_succeeds(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHello.\n"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) < 3:
            return _proc(returncode=1, stderr="HTTP Error 429: Too Many Requests")
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    sleeps = []
    transcript = fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc",
        run=fake_run,
        workdir=tmp_path,
        sleep=sleeps.append,
    )
    assert len(calls) == 3
    assert transcript.full_text() == "Hello."
    # Exponential backoff: 2s, then 4s — no real waiting since sleep is faked.
    assert sleeps == [2.0, 4.0]


@pytest.mark.unit
def test_fetch_video_transcript_raises_youtube_fetch_error_on_persistent_429(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _proc(returncode=1, stderr="HTTP Error 429: Too Many Requests")

    with pytest.raises(YoutubeFetchError):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=stillbad",
            run=fake_run,
            workdir=tmp_path,
            sleep=lambda seconds: None,
        )
    # Bounded: capped attempts, not retried forever.
    assert len(calls) == 3


@pytest.mark.unit
def test_fetch_video_transcript_does_not_retry_non_transient_failure(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _proc(returncode=1, stderr="Video unavailable")

    with pytest.raises(YoutubeFetchError, match="Video unavailable"):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=gone",
            run=fake_run,
            workdir=tmp_path,
            sleep=lambda seconds: (_ for _ in ()).throw(AssertionError("should not sleep")),
        )
    assert len(calls) == 1


@pytest.mark.unit
def test_list_playlist_video_urls_raises_youtube_fetch_error_on_persistent_5xx():
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _proc(returncode=1, stderr="HTTP Error 503: Service Unavailable")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls(
            "https://www.youtube.com/playlist?list=bad",
            run=fake_run,
            sleep=lambda seconds: None,
        )
    assert len(calls) == 3


# ---- Phase 20: DISTIL_YOUTUBE_API_KEY wiring (innertube_host/innertube_key) ----


@pytest.mark.unit
def test_list_playlist_video_urls_omits_key_args_when_env_unset(monkeypatch):
    monkeypatch.delenv("DISTIL_YOUTUBE_API_KEY", raising=False)
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android,web"
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_list_playlist_video_urls_passes_api_key_when_env_set(monkeypatch):
    monkeypatch.setenv("DISTIL_YOUTUBE_API_KEY", "secret-key-123")
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        value = cmd[idx + 1]
        assert value.startswith("youtube:player_client=android,web;")
        assert "innertube_host=youtubei.googleapis.com" in value
        assert "innertube_key=secret-key-123" in value
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_fetch_video_transcript_omits_key_args_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTIL_YOUTUBE_API_KEY", raising=False)
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHello.\n"

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android,web"
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


@pytest.mark.unit
def test_fetch_video_transcript_passes_api_key_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_YOUTUBE_API_KEY", "secret-key-123")
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHello.\n"

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        value = cmd[idx + 1]
        assert value.startswith("youtube:player_client=android,web;")
        assert "innertube_host=youtubei.googleapis.com" in value
        assert "innertube_key=secret-key-123" in value
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


# ---- T-Y6: malformed playlist JSON -> clear error, not a crash ----


@pytest.mark.unit
def test_list_playlist_video_urls_raises_on_bad_json():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0, stdout="not json")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=weird", run=fake_run)
