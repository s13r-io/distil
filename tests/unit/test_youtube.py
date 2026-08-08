"""Phase 1 — youtube.py: fetch captions/enumerate playlists via yt-dlp. Tests T-Y1..Y6.

``yt-dlp`` is invoked through an injected ``run`` callable (``subprocess.run`` signature) so
these stay unit tests: no network, no subprocess, no real binary required.
"""

from __future__ import annotations

import json
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
def test_list_playlist_video_urls_raises_on_yt_dlp_failure():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr="playlist does not exist")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=bad", run=fake_run)


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
        (tmp_path / f"{out_prefix.split('/')[-1]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    transcript = fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path
    )
    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Welcome to the talk."
    assert transcript.segments[0].timestamp == "00:00:01"


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


# ---- T-Y6: malformed playlist JSON -> clear error, not a crash ----


@pytest.mark.unit
def test_list_playlist_video_urls_raises_on_bad_json():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0, stdout="not json")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=weird", run=fake_run)
