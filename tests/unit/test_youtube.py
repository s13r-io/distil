"""Phase 1 — youtube.py: fetch captions/enumerate playlists via yt-dlp. Tests T-Y1..Y6.

``yt-dlp`` is invoked through an injected ``run`` callable (``subprocess.run`` signature) so
these stay unit tests: no network, no subprocess, no real binary required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from distil.ingest import Transcript, TranscriptTooShortError
from distil.youtube import (
    PotDiagnostic,
    YoutubeFetchError,
    _detect_browser_session,
    _redact_pot_diagnostic,
    _surface_error,
    diagnose_pot,
    fetch_raw_captions,
    fetch_video_transcript,
    is_bot_check_refusal,
    is_playlist_url,
    list_playlist_video_urls,
)


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# A single-cue caption body long enough to clear ingest.py's 50-word floor (fetch_video_transcript
# parses via ingest_srt_text) — used by tests that don't care about the caption text itself.
_LONG_CAPTION_TEXT = (
    "Hello and welcome to this talk about writing code that your whole team can "
    "actually read, understand, and safely change without breaking something else. "
    "We will cover naming, function size, and how to structure a codebase so that "
    "new contributors can find their way around it without needing a long guided tour."
)
_LONG_SRT_BODY = f"1\n00:00:01,000 --> 00:00:03,000\n{_LONG_CAPTION_TEXT}\n"


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
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android_vr,web_safari,mweb"
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
        "2\n00:00:04,000 --> 00:00:06,000\n"
        + " ".join(["Let's get started with the rest of it."] * 8)
        + "\n"
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
def test_fetch_video_transcript_reports_transcript_fetch_and_caption_parse_phases(tmp_path):
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    events: list[tuple[str, str]] = []
    fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path,
        on_phase=lambda phase, event: events.append((phase, event)),
    )
    assert events == [
        ("transcript_fetch", "start"), ("transcript_fetch", "finish"),
        ("caption_parse", "start"), ("caption_parse", "finish"),
    ]


@pytest.mark.unit
def test_fetch_video_transcript_reports_only_fetch_start_on_yt_dlp_failure(tmp_path):
    """A stall/failure during the network call must surface as a stuck 'transcript_fetch',
    never silently advance to caption_parse."""
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr="ERROR: Sign in to confirm you're not a bot")

    events: list[tuple[str, str]] = []
    with pytest.raises(YoutubeFetchError):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path,
            on_phase=lambda phase, event: events.append((phase, event)),
        )
    assert events == [("transcript_fetch", "start")]


@pytest.mark.unit
def test_fetch_video_transcript_passes_player_client_fallback_chain(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android_vr,web_safari,mweb"
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
    stale.write_text("1\n00:00:01,000 --> 00:00:02,000\nSTALE OLD CAPTION\n", encoding="utf-8")

    fresh_text = "Fresh caption for this fetch, " + "long enough to clear the word floor. " * 8
    fresh_srt = f"1\n00:00:01,000 --> 00:00:03,000\n{fresh_text.strip()}\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(fresh_srt, encoding="utf-8")
        return _proc(returncode=0)

    transcript = fetch_video_transcript(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path
    )
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text.startswith("Fresh caption for this fetch")
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


# ---- Owner-trust: a too-short fetched transcript is rejected, distinctly from a fetch failure --


@pytest.mark.unit
def test_fetch_video_transcript_raises_transcript_too_short_not_wrapped_as_fetch_error(tmp_path):
    """The fetch itself succeeded and the captions parsed fine — there just isn't enough of
    them. This must surface as TranscriptTooShortError, never folded into YoutubeFetchError,
    so callers can tell "too short" apart from a genuine fetch/parse failure."""
    srt_body = "1\n00:00:01,000 --> 00:00:02,000\nToo short.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    with pytest.raises(TranscriptTooShortError):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=short", run=fake_run, workdir=tmp_path
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
    srt_body = _LONG_SRT_BODY
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
    assert transcript.full_text() == _LONG_CAPTION_TEXT
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


# ---- Phase 22: DISTIL_POT_PROVIDER_URL wiring (youtubepot-bgutilhttp:base_url) ----
# Replaces the removed Phase 20 DISTIL_YOUTUBE_API_KEY mechanism (innertube_host/innertube_key
# could never address a bot-identity challenge — see distil/youtube.py's module docstring).


@pytest.mark.unit
def test_list_playlist_video_urls_omits_pot_provider_args_when_env_unset(monkeypatch):
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        # Byte-identical to pre-Phase-22 behavior: exactly one --extractor-args pair.
        assert cmd.count("--extractor-args") == 1
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android_vr,web_safari,mweb"
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_list_playlist_video_urls_passes_pot_provider_url_when_env_set(monkeypatch):
    monkeypatch.setenv(
        "DISTIL_POT_PROVIDER_URL", "http://bgutil-pot-provider.railway.internal:4416"
    )
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        assert cmd.count("--extractor-args") == 2
        first = cmd.index("--extractor-args")
        assert cmd[first + 1] == "youtube:player_client=android_vr,web_safari,mweb;fetch_pot=always"
        second = cmd.index("--extractor-args", first + 1)
        assert cmd[second + 1] == (
            "youtubepot-bgutilhttp:base_url=http://bgutil-pot-provider.railway.internal:4416"
        )
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_fetch_video_transcript_omits_pot_provider_args_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        assert cmd.count("--extractor-args") == 1
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android_vr,web_safari,mweb"
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


@pytest.mark.unit
def test_fetch_video_transcript_passes_pot_provider_url_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "DISTIL_POT_PROVIDER_URL", "http://bgutil-pot-provider.railway.internal:4416"
    )
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        assert cmd.count("--extractor-args") == 2
        first = cmd.index("--extractor-args")
        assert cmd[first + 1] == "youtube:player_client=android_vr,web_safari,mweb;fetch_pot=always"
        second = cmd.index("--extractor-args", first + 1)
        assert cmd[second + 1] == (
            "youtubepot-bgutilhttp:base_url=http://bgutil-pot-provider.railway.internal:4416"
        )
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


# ---- Phase 23: fetch_pot=always — the actual fix. yt-dlp's default `fetch_pot=auto` policy
# never asks *any* provider for a player-context token on our client chain (see the module
# docstring's Phase 23 paragraph) — `fetch_pot=always` forces the attempt. Only added when a
# provider is configured; must live in the *same* youtube: extractor-args value as player_client,
# never a separate --extractor-args pair, since yt-dlp replaces (not merges) repeated
# --extractor-args for one extractor. ----


@pytest.mark.unit
def test_list_playlist_video_urls_omits_fetch_pot_when_env_unset(monkeypatch):
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--extractor-args")
        assert cmd[idx + 1] == "youtube:player_client=android_vr,web_safari,mweb"
        assert "fetch_pot" not in cmd[idx + 1]
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_fetch_video_transcript_folds_fetch_pot_always_into_youtube_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "DISTIL_POT_PROVIDER_URL", "http://bgutil-pot-provider.railway.internal:4416"
    )
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        # Exactly one youtube: pair (never two — the second would silently discard the first,
        # dropping player_client entirely).
        youtube_pairs = [
            cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--extractor-args" and cmd[i + 1].startswith("youtube:")
        ]
        assert youtube_pairs == ["youtube:player_client=android_vr,web_safari,mweb;fetch_pot=always"]
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


# ---- Phase 23: diagnose_pot — the permanent diagnostic capability, and its redaction. ----


@pytest.mark.unit
def test_diagnose_pot_parses_provider_discovery_and_context_attempts(monkeypatch):
    monkeypatch.delenv("DISTIL_POT_PROVIDER_URL", raising=False)
    verbose_output = (
        "[debug] Command-line config: ['-v', ...]\n"
        "[youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external)\n"
        "[youtube] abc12345678: Downloading webpage\n"
        "[youtubepot-bgutilhttp] [debug] Generating a player PO Token for web_safari client "
        "via bgutil HTTP server\n"
        "[youtubepot-bgutilhttp] [debug] Generating a subs PO Token for web_safari client "
        "via bgutil HTTP server\n"
        "WARNING: No title found in player responses; falling back to title from initial data\n"
    )

    def fake_run(cmd, **kwargs):
        assert "-v" in cmd
        assert "--simulate" in cmd
        return _proc(returncode=0, stdout=verbose_output, stderr="")

    result = diagnose_pot("https://www.youtube.com/watch?v=abc12345678", run=fake_run)
    assert isinstance(result, PotDiagnostic)
    assert result.returncode == 0
    assert result.provider_discovery == "[youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external)"
    assert ("player", "web_safari") in result.context_attempts
    assert ("subs", "web_safari") in result.context_attempts


@pytest.mark.unit
def test_diagnose_pot_reports_no_attempts_when_never_asked():
    # Finding (a) from the task this closes: no provider line, no context attempts at all.
    verbose_output = (
        "[youtube] abc12345678: Downloading webpage\n"
        "ERROR: [youtube] abc12345678: Sign in to confirm you're not a bot\n"
    )

    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stdout="", stderr=verbose_output)

    result = diagnose_pot("https://www.youtube.com/watch?v=abc12345678", run=fake_run)
    assert result.provider_discovery is None
    assert result.context_attempts == []
    assert "Sign in to confirm you're not a bot" in result.raw_output
    assert result.bot_check_detected is True


# ---- Phase 24: diagnose_pot surfaces whether the bot-check safety net is alive ----


@pytest.mark.unit
def test_diagnose_pot_bot_check_detected_true_on_real_curly_apostrophe_refusal():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stdout="", stderr=_REAL_BOT_CHECK_STDERR)

    result = diagnose_pot("https://www.youtube.com/watch?v=abc12345678", run=fake_run)
    assert result.bot_check_detected is True


@pytest.mark.unit
def test_diagnose_pot_bot_check_detected_false_when_run_succeeds():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0, stdout="[info] abc: Downloading subtitles: en\n", stderr="")

    result = diagnose_pot("https://www.youtube.com/watch?v=abc12345678", run=fake_run)
    assert result.bot_check_detected is False


@pytest.mark.unit
def test_diagnose_pot_uses_pot_trace_and_extractor_args(monkeypatch):
    monkeypatch.setenv(
        "DISTIL_POT_PROVIDER_URL", "http://bgutil-pot-provider.railway.internal:4416"
    )

    def fake_run(cmd, **kwargs):
        youtube_pairs = [
            cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--extractor-args" and cmd[i + 1].startswith("youtube:")
        ]
        # One combined youtube: value carrying player_client, fetch_pot, and pot_trace.
        assert youtube_pairs == [
            "youtube:player_client=android_vr,web_safari,mweb;fetch_pot=always;pot_trace=true"
        ]
        assert "--simulate" in cmd
        return _proc(returncode=0, stdout="", stderr="")

    diagnose_pot("https://www.youtube.com/watch?v=abc", run=fake_run)


@pytest.mark.unit
def test_redact_pot_diagnostic_strips_token_value_and_provider_url():
    text = (
        "[youtubepot-bgutilhttp] [trace] Generated POT: super-secret-token-value-123\n"
        "PO Token response from \"bgutil:http\" provider: "
        "PoTokenResponse(po_token='another-secret-abc', expires_at=1234567890)\n"
        "extractor-args: youtubepot-bgutilhttp:base_url=http://bgutil-pot-provider.railway.internal:4416\n"
    )
    redacted = _redact_pot_diagnostic(
        text, "http://bgutil-pot-provider.railway.internal:4416"
    )
    assert "super-secret-token-value-123" not in redacted
    assert "another-secret-abc" not in redacted
    assert "bgutil-pot-provider.railway.internal" not in redacted
    assert "<redacted>" in redacted
    assert "<redacted-provider-url>" in redacted


@pytest.mark.unit
def test_diagnose_pot_output_never_leaks_provider_url(monkeypatch):
    provider_url = "http://bgutil-pot-provider.railway.internal:4416"
    monkeypatch.setenv("DISTIL_POT_PROVIDER_URL", provider_url)

    def fake_run(cmd, **kwargs):
        # yt-dlp itself would echo the extractor-args it was given in -v output.
        return _proc(
            returncode=0,
            stdout=f"[debug] Extractor Args: youtubepot-bgutilhttp:base_url={provider_url}\n",
        )

    result = diagnose_pot("https://www.youtube.com/watch?v=abc", run=fake_run)
    assert provider_url not in result.raw_output


@pytest.mark.unit
def test_diagnose_pot_never_raises_and_redacts_url_on_timeout(monkeypatch):
    provider_url = "http://bgutil-pot-provider.railway.internal:4416"
    monkeypatch.setenv("DISTIL_POT_PROVIDER_URL", provider_url)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60.0)

    result = diagnose_pot("https://www.youtube.com/watch?v=abc", run=fake_run, timeout=60.0)
    assert isinstance(result, PotDiagnostic)
    assert result.returncode != 0
    assert result.provider_discovery is None
    assert result.context_attempts == []
    assert provider_url not in result.raw_output


@pytest.mark.unit
def test_no_distil_youtube_api_key_handling_survives():
    import distil.youtube as youtube_module

    source = Path(youtube_module.__file__).read_text(encoding="utf-8")
    # Historical mentions in comments/docstrings explaining *why* it was removed are fine; no
    # code should read the env var itself.
    assert 'os.environ.get("DISTIL_YOUTUBE_API_KEY")' not in source
    assert "innertube_key" not in source
    assert "innertube_host" not in source


# ---- T-Y6: malformed playlist JSON -> clear error, not a crash ----


@pytest.mark.unit
def test_list_playlist_video_urls_raises_on_bad_json():
    def fake_run(cmd, **kwargs):
        return _proc(returncode=0, stdout="not json")

    with pytest.raises(YoutubeFetchError):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=weird", run=fake_run)


# ---- Phase 21: the real yt-dlp error must survive warning-heavy stderr, and the complete
# stderr must reach the logs — regression coverage for the `_tail`-returned-the-head bug. ----

# Mirrors what real yt-dlp emits: the SABR-format warning plus the 90-day staleness warning,
# both non-fatal, together comfortably exceeding the old head-truncation's 300-char budget
# before the actual `ERROR:` line ever appears.
_WARNING_NOISE = (
    "WARNING: [youtube] abc12345678: Some android client https formats have been skipped as "
    "they are missing a URL. YouTube may have enabled the SABR-only streaming experiment for "
    "the current session. See https://github.com/yt-dlp/yt-dlp/issues/12482 for more details\n"
    "WARNING: Your yt-dlp version (2024.08.06) is older than 90 days! It is strongly "
    'recommended to always use the latest version. Run "yt-dlp --update" or "yt-dlp -U" to '
    "update. To suppress this warning, add --no-update to your command/config.\n"
)


def test_warning_noise_exceeds_old_truncation_budget():
    # Sanity check on the fixture itself: this is exactly the shape of stderr that defeated the
    # old head-truncating `_tail(text, limit=300)` helper.
    assert len(_WARNING_NOISE) > 300


@pytest.mark.unit
def test_surface_error_prefers_error_line_over_leading_warnings():
    stderr = (
        _WARNING_NOISE + "ERROR: [youtube] abc12345678: Video unavailable. This video is private.\n"
    )
    assert (
        _surface_error(stderr)
        == "ERROR: [youtube] abc12345678: Video unavailable. This video is private."
    )


@pytest.mark.unit
def test_surface_error_falls_back_to_genuine_tail_without_error_line():
    # No `ERROR:` line at all (e.g. a bare crash message) — fall back to the *last* `limit`
    # characters, not the first, since head-truncation is exactly the bug being fixed.
    stderr = ("noise " * 100) + "the actually useful bit at the end"
    result = _surface_error(stderr, limit=50)
    assert result == stderr[-50:]
    assert "the actually useful bit at the end" in result


@pytest.mark.unit
def test_fetch_video_transcript_surfaces_error_past_warning_noise(tmp_path):
    stderr = (
        _WARNING_NOISE + "ERROR: [youtube] abc12345678: Video unavailable. This video is private.\n"
    )

    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr=stderr)

    with pytest.raises(YoutubeFetchError, match="Video unavailable"):
        fetch_video_transcript(
            "https://www.youtube.com/watch?v=abc12345678", run=fake_run, workdir=tmp_path
        )


@pytest.mark.unit
def test_list_playlist_video_urls_surfaces_error_past_warning_noise():
    stderr = _WARNING_NOISE + "ERROR: [youtube:tab] Playlist does not exist.\n"

    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr=stderr)

    with pytest.raises(YoutubeFetchError, match="Playlist does not exist"):
        list_playlist_video_urls("https://www.youtube.com/playlist?list=bad", run=fake_run)


@pytest.mark.unit
def test_fetch_video_transcript_logs_complete_untruncated_stderr(tmp_path, caplog):
    stderr = (
        _WARNING_NOISE + "ERROR: [youtube] abc12345678: Video unavailable. This video is private.\n"
    )

    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr=stderr)

    with caplog.at_level("ERROR", logger="distil.youtube"):
        with pytest.raises(YoutubeFetchError):
            fetch_video_transcript(
                "https://www.youtube.com/watch?v=abc12345678", run=fake_run, workdir=tmp_path
            )
    assert stderr in caplog.text


@pytest.mark.unit
def test_list_playlist_video_urls_logs_complete_untruncated_stderr(caplog):
    stderr = _WARNING_NOISE + "ERROR: [youtube:tab] Playlist does not exist.\n"

    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr=stderr)

    with caplog.at_level("ERROR", logger="distil.youtube"):
        with pytest.raises(YoutubeFetchError):
            list_playlist_video_urls("https://www.youtube.com/playlist?list=bad", run=fake_run)
    assert stderr in caplog.text


@pytest.mark.unit
def test_fetch_video_transcript_passes_no_update_flag(tmp_path):
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        assert "--no-update" in cmd
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


@pytest.mark.unit
def test_list_playlist_video_urls_passes_no_update_flag():
    payload = json.dumps({"entries": [{"id": "abc"}]})

    def fake_run(cmd, **kwargs):
        assert "--no-update" in cmd
        return _proc(returncode=0, stdout=payload)

    list_playlist_video_urls("https://www.youtube.com/playlist?list=PL1", run=fake_run)


@pytest.mark.unit
def test_fetch_video_transcript_requests_srt_natively_without_convert_subs(tmp_path):
    srt_body = _LONG_SRT_BODY

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--sub-format")
        assert cmd[idx + 1] == "srt/best"
        assert "--convert-subs" not in cmd
        out_index = cmd.index("-o") + 1
        out_prefix = cmd[out_index]
        Path(f"{out_prefix}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_video_transcript("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


# ---- bot-check refusal detection (collector queue) ----
#
# Phase 24 incident: the original `_BOT_CHECK_MARKER` literal and its test fixtures were both
# typed with a straight U+0027 apostrophe ("you're"), but YouTube's actual refusal text (verified
# from production failures for videos AbpyqAfxZ8c and QER-0DaC-Gk) uses the curly U+2019 RIGHT
# SINGLE QUOTATION MARK ("you’re"). Test and implementation shared the same wrong assumption,
# so the suite was green while detection matched nothing in production. `_REAL_BOT_CHECK_STDERR`
# below is that genuine captured wording, curly apostrophe included byte-for-byte, and is what the
# regression test below is built from — not a hand-retyped paraphrase.

_REAL_BOT_CHECK_STDERR = (
    "ERROR: [youtube] AbpyqAfxZ8c: Sign in to confirm you’re not a bot. "
    "This helps protect our community. Learn more"
)


@pytest.mark.unit
def test_is_bot_check_refusal_true_for_real_captured_curly_apostrophe_output():
    """Regression test for the Phase 24 incident: fails against the pre-fix straight-quote
    literal match, passes once detection tolerates YouTube's actual curly apostrophe."""
    exc = YoutubeFetchError(f"yt-dlp failed: {_REAL_BOT_CHECK_STDERR}")
    assert is_bot_check_refusal(exc) is True


@pytest.mark.unit
def test_is_bot_check_refusal_true_for_straight_apostrophe_variant():
    # Covered in case YouTube ever reverts to (or some yt-dlp version normalizes to) a plain
    # ASCII apostrophe — the original, pre-incident wording must keep matching too.
    exc = YoutubeFetchError(
        "yt-dlp failed: ERROR: [youtube] abc12345678: Sign in to confirm you're not a bot"
    )
    assert is_bot_check_refusal(exc) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "apostrophe",
    [
        "'",  # U+0027 APOSTROPHE (straight — pre-incident fixture wording)
        "’",  # RIGHT SINGLE QUOTATION MARK (curly — what YouTube actually sends)
        "‘",  # LEFT SINGLE QUOTATION MARK
        "ʼ",  # MODIFIER LETTER APOSTROPHE
        "`",  # GRAVE ACCENT, sometimes substituted for an apostrophe
        "",  # no apostrophe at all — a plausible future reword ("you not a bot"->"youre")
    ],
)
def test_is_bot_check_refusal_tolerates_apostrophe_typographic_variants(apostrophe):
    exc = f"Sign in to confirm you{apostrophe}re not a bot"
    assert is_bot_check_refusal(exc) is True


@pytest.mark.unit
def test_is_bot_check_refusal_is_case_insensitive():
    exc = "sign in to confirm YOU’RE not a bot"
    assert is_bot_check_refusal(exc) is True


@pytest.mark.unit
def test_is_bot_check_refusal_false_for_no_captions():
    exc = YoutubeFetchError("No English captions available for this video.")
    assert is_bot_check_refusal(exc) is False


@pytest.mark.unit
def test_is_bot_check_refusal_false_for_playlist_listing_failure():
    exc = YoutubeFetchError("Could not list playlist: playlist does not exist")
    assert is_bot_check_refusal(exc) is False


@pytest.mark.unit
def test_is_bot_check_refusal_accepts_a_plain_string_too():
    assert is_bot_check_refusal(_REAL_BOT_CHECK_STDERR) is True


# ---- Helper 2 (collector): fetch_raw_captions shares _fetch_captions_raw with _fetch_into ----


@pytest.mark.unit
def test_fetch_raw_captions_returns_unparsed_srt_text(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nWelcome to the talk.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    raw = fetch_raw_captions("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)
    assert raw == srt_body


@pytest.mark.unit
def test_fetch_raw_captions_raises_on_yt_dlp_failure(tmp_path):
    def fake_run(cmd, **kwargs):
        return _proc(returncode=1, stderr="Video unavailable")

    with pytest.raises(YoutubeFetchError, match="Video unavailable"):
        fetch_raw_captions("https://www.youtube.com/watch?v=gone", run=fake_run, workdir=tmp_path)


@pytest.mark.unit
def test_fetch_raw_captions_passes_cookies_from_browser_to_yt_dlp(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"
    seen_cmds = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_raw_captions(
        "https://www.youtube.com/watch?v=abc",
        run=fake_run,
        workdir=tmp_path,
        cookies_from_browser="chrome",
    )
    cmd = seen_cmds[0]
    idx = cmd.index("--cookies-from-browser")
    assert cmd[idx + 1] == "chrome"
    assert "--cookies" in cmd


@pytest.mark.unit
def test_fetch_raw_captions_never_passes_cookies_flags_when_unset(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"

    def fake_run(cmd, **kwargs):
        assert "--cookies-from-browser" not in cmd
        assert "--cookies" not in cmd
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    fetch_raw_captions("https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path)


@pytest.mark.unit
def test_fetch_raw_captions_reports_signed_in_session(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        cookies_index = cmd.index("--cookies") + 1
        Path(cmd[cookies_index]).write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t0\tCONSENT\tYES+1\n"
            "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSAPISID\tsecret-value\n",
            encoding="utf-8",
        )
        return _proc(returncode=0)

    sessions = []
    fetch_raw_captions(
        "https://www.youtube.com/watch?v=abc",
        run=fake_run,
        workdir=tmp_path,
        cookies_from_browser="chrome",
        on_session=sessions.append,
    )
    assert sessions == ["signed_in"]


@pytest.mark.unit
def test_fetch_raw_captions_reports_anonymous_session(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        cookies_index = cmd.index("--cookies") + 1
        Path(cmd[cookies_index]).write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t0\tCONSENT\tYES+1\n"
            ".youtube.com\tTRUE\t/\tTRUE\t0\tYSC\tabc\n",
            encoding="utf-8",
        )
        return _proc(returncode=0)

    sessions = []
    fetch_raw_captions(
        "https://www.youtube.com/watch?v=abc",
        run=fake_run,
        workdir=tmp_path,
        cookies_from_browser="chrome",
        on_session=sessions.append,
    )
    assert sessions == ["anonymous"]


@pytest.mark.unit
def test_fetch_raw_captions_reports_unknown_session_when_no_jar_written(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"

    def fake_run(cmd, **kwargs):
        # Simulates a yt-dlp version/config that never wrote the cookie jar at all.
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    sessions = []
    fetch_raw_captions(
        "https://www.youtube.com/watch?v=abc",
        run=fake_run,
        workdir=tmp_path,
        cookies_from_browser="chrome",
        on_session=sessions.append,
    )
    assert sessions == ["unknown"]


@pytest.mark.unit
def test_fetch_raw_captions_never_calls_on_session_when_cookies_from_browser_unset(tmp_path):
    srt_body = "1\n00:00:01,000 --> 00:00:03,000\nHi.\n"

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        Path(f"{cmd[out_index]}.en.srt").write_text(srt_body, encoding="utf-8")
        return _proc(returncode=0)

    sessions = []
    fetch_raw_captions(
        "https://www.youtube.com/watch?v=abc", run=fake_run, workdir=tmp_path,
        on_session=sessions.append,
    )
    assert sessions == []


@pytest.mark.unit
def test_fetch_raw_captions_reports_session_even_when_fetch_ultimately_fails(tmp_path):
    def fake_run(cmd, **kwargs):
        cookies_index = cmd.index("--cookies") + 1
        Path(cmd[cookies_index]).write_text(
            "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tLOGIN_INFO\tsecret-value\n",
            encoding="utf-8",
        )
        return _proc(returncode=1, stderr="Sign in to confirm you're not a bot")

    sessions = []
    with pytest.raises(YoutubeFetchError):
        fetch_raw_captions(
            "https://www.youtube.com/watch?v=abc",
            run=fake_run,
            workdir=tmp_path,
            cookies_from_browser="chrome",
            on_session=sessions.append,
        )
    assert sessions == ["signed_in"]


@pytest.mark.unit
def test_detect_browser_session_deletes_the_cookie_file_after_reading(tmp_path):
    jar = tmp_path / "cookies.txt"
    jar.write_text(".youtube.com\tTRUE\t/\tTRUE\t0\tYSC\tabc\n", encoding="utf-8")
    assert _detect_browser_session(jar) == "anonymous"
    assert not jar.exists()


@pytest.mark.unit
def test_detect_browser_session_unknown_when_jar_missing(tmp_path):
    assert _detect_browser_session(tmp_path / "missing-cookies.txt") == "unknown"
