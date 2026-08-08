"""YouTube fetch (Phase 1). ARCHITECTURE.md source-ingest extension; TESTING T-Y*.

Wraps ``yt-dlp`` to (a) enumerate the videos in a playlist and (b) fetch English captions for
one video and parse them into a :class:`~distil.ingest.Transcript` via ``ingest.py``'s existing
SRT parser — so downstream stages (triage/extract/normalize/note) never know the source was
YouTube rather than an uploaded ``.srt``. ``yt-dlp`` is invoked through an injectable ``run``
callable (``subprocess.run`` signature) so callers can fake the process boundary in tests.

Both call sites pin ``--extractor-args youtube:player_client=android_vr,web_safari,mweb`` (Phase
23 added ``mweb``; Phase 21 was ``android_vr,web_safari``; Phase 19 originally pinned
``android,web``). ``android`` was dropped from yt-dlp's own default client list years ago and is
what produces the "SABR-only streaming experiment" warning on today's yt-dlp; ``android_vr`` is
its currently-recommended, still-unauthenticated replacement — it's yt-dlp's own
``_DEFAULT_JSLESS_CLIENTS`` (see ``yt_dlp.extractor.youtube._video``), i.e. what yt-dlp itself
picks when no JS runtime is available, which is exactly our situation: ``python:3.11-slim`` (the
Dockerfile base) has none, and android_vr's ``REQUIRE_JS_PLAYER`` is ``False``, needing no
interpreter to solve the n-parameter signature challenge that the plain ``web`` client would.
``web_safari`` — yt-dlp's other default-chain member (its full ``_DEFAULT_CLIENTS``) — stays as a
fallback in the same call for the rare video whose android_vr response lacks the caption track;
captions don't need the JS-gated format-signature step, so pairing it with a JS-requiring fallback
is safe even with no JS runtime installed. ``mweb`` (Phase 23) is a third-tier fallback added
*specifically* to give a PO token somewhere to attach to — see that phase's paragraph below for
why ``android_vr``/``web_safari`` alone can never receive one, no matter how the provider is
configured. Re-derive this chain from ``yt_dlp.extractor.youtube._video``'s
``_DEFAULT_CLIENTS``/``_DEFAULT_JSLESS_CLIENTS``/``_WEBPAGE_CLIENTS`` next time it needs
revisiting — yt-dlp changes its recommended clients every few months as YouTube reacts, and continue
to weigh any addition against the no-JS-runtime constraint the way Phase 23 did (below), not just
against caption support.

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

**Phase 23: the provider working once and then not, root-caused.** The Phase 22 wiring shipped
byte-identical to the description above, but a token was never even *requested* for most videos —
verified against yt-dlp 2026.7.4's own client/policy tables and a live (non-datacenter) run with
the plugin installed and ``pot_trace`` on, not inferred from production logs. Two independent gaps
stacked:

1. yt-dlp's PO token fetch defaults to ``fetch_pot=auto`` (see its ``--extractor-args`` help):
   under ``auto``, yt-dlp only asks *any* provider for a token in a given context (player / gvs /
   subs) if that client's static ``PLAYER_PO_TOKEN_POLICY`` / ``GVS_PO_TOKEN_POLICY`` /
   ``SUBS_PO_TOKEN_POLICY`` marks it ``required`` or ``recommended`` — otherwise
   ``_fetch_po_token`` returns ``None`` before any provider is ever consulted
   (``yt_dlp.extractor.youtube._video``). In the pinned version *every* WebPO-family client has
   ``PLAYER_PO_TOKEN_POLICY(required=False, recommended=False)`` — player-context tokens are
   never auto-requested for any client a provider could serve. The subs context worked once
   because it *isn't* purely policy-gated: a caption track URL can carry an ``xpe``/``xpv``
   experiment flag that forces a subs-token fetch dynamically, independent of the (also
   ``False``/``False``) static subs policy — that flag's presence per video/session is what made
   one video's provider log a full generate-and-fetch flow while the next showed nothing.
2. Deeper and specific to *this* chain: ``web_safari`` is yt-dlp's ``_DEFAULT_WEBPAGE_CLIENT``
   (``_WEBPAGE_CLIENTS = ('web', 'web_safari')``), so when it's also the client used for the
   watch-page HTML scrape, ``_extract_player_responses`` reuses that page's *already-embedded*
   player response for it (``pr = initial_pr``) instead of making a dedicated Innertube ``/player``
   call — and explicitly skips the player-token fetch in that case (see its own
   ``# Don't need a player PO token for WEB if using player response from webpage`` comment). A
   client with no dedicated Innertube player call structurally has no player-context request for a
   token to attach to, `fetch_pot` setting notwithstanding — confirmed live: forcing
   ``fetch_pot=always`` produced zero ``player`` PO-token attempts for ``web_safari``, only
   ``gvs`` (a real per-format token pull that already worked pre-fix, since web_safari's static GVS
   policy already says ``required``). ``android_vr`` fares no better for the opposite reason: it
   does make its own dedicated Innertube call, but isn't a WebPO client at all
   (``bgutil-ytdlp-pot-provider``'s ``_SUPPORTED_CLIENTS`` is exactly
   ``yt_dlp.extractor.youtube.pot.utils.WEBPO_CLIENTS``, which excludes it), so the provider
   rejects it outright regardless of policy. Net effect: within the pre-Phase-23 chain, *no*
   client could ever receive a genuine player-context token — not a misconfiguration on distil's
   part, since ``android_vr,web_safari`` is yt-dlp's own current ``_DEFAULT_CLIENTS``.

This is finding (a) from the originating task brief ("never asked"), confirmed from source and a
live run rather than assumed. Both gaps needed closing, and both are yt-dlp's own documented
mechanisms, not guesses:

- When ``DISTIL_POT_PROVIDER_URL`` is set, :func:`_extractor_args` folds ``fetch_pot=always`` into
  the *same* ``youtube:`` value as ``player_client`` (joined with ``;`` — yt-dlp resolves repeated
  ``--extractor-args`` for one extractor by *replacing* the whole value, not merging, so this must
  live in one string, never a separate ``--extractor-args`` pair). This forces the attempt for
  every context on every call site regardless of static policy, while remaining a no-op (a fast
  local rejection, no network call) for ``android_vr``, which the provider was never going to
  serve anyway. GVS is irrelevant here and deliberately not chased further: both call sites pass
  ``--skip-download``, so the streaming-format code path is the only thing that ever needed it,
  and it already worked.
- ``mweb`` was added as a third-tier fallback specifically because it's WebPO-eligible *and* is
  never a ``_WEBPAGE_CLIENTS`` member, so it always makes its own dedicated Innertube player call —
  giving a player-context token somewhere to actually attach. (``tv``/``tv_simply``/``web_creator``
  share that property but were passed over: ``web_music`` is directly flagged broken for bgutil
  token generation in the plugin's own source, and the others are documented as special-purpose
  age-gate/embed workarounds rather than general-purpose fallbacks; ``mweb`` is yt-dlp's
  general-purpose mobile-web client, and is explicitly called out alongside plain ``web`` as one of
  the two clients whose player response actually populates ``translationLanguages`` — i.e. it's a
  well-supported choice for caption-related extraction specifically, not an arbitrary pick.)
  Verified from a live, non-datacenter run with ``deno``/``node`` removed from ``PATH`` (matching
  the Dockerfile's no-JS-runtime image exactly): a real, unauthenticated caption fetch through this
  chain still succeeds end-to-end with no provider configured (behavior-preserving baseline), and
  with a provider configured, ``fetch_pot=always`` now produces a genuine
  ``Generating a player PO Token for mweb client via bgutil HTTP server`` attempt — the exact
  request class that was previously impossible to reach. ``mweb``'s ``REQUIRE_JS_PLAYER=True``
  only gates *format*/nsig resolution, never the player-response or caption-track extraction this
  module needs, so this doesn't reopen the no-JS-runtime constraint discussed above; if a future
  client swap ever changes that, treat it exactly as seriously as adding a JS runtime would be.

None of this is a guarantee — see the Phase 22 paragraph above. What's proven is that a
player-context PO token attempt now reaches the provider where before it structurally could not;
whether that specific token clears YouTube's datacenter-IP bot check can only be confirmed on the
actual deploy, never from this (or any) residential dev machine.

:func:`diagnose_pot` is the permanent fix for the diagnostic gap that cost two round trips to
learn this: a verbose, ``--simulate`` yt-dlp run (CLI: ``distil youtube-diagnose-pot <url>``; web:
``GET /diagnostics/youtube-pot?url=...``) that surfaces yt-dlp's own ``PO Token Providers:``
discovery line and every context/client pair it actually attempted a fetch for, from the running
service itself, with no shell access to the container required.

Speech-to-text (Whisper) for uncaptioned videos is out of scope; those videos raise
:class:`YoutubeFetchError` so callers can skip + report them without failing a whole playlist.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from distil.ingest import IngestError, Transcript, ingest_srt_text
from distil.source import is_youtube_host

_logger = logging.getLogger(__name__)

_YT_DLP = "yt-dlp"

# android_vr needs no JS runtime (yt-dlp's own no-JS default; see module docstring) and is
# throttled far less than web on server IPs; web_safari is a fallback in the same invocation for
# the rare video whose android_vr response omits captions; mweb (Phase 23) is a further fallback
# that's both PO-token-eligible and never reuses the webpage's embedded player response, so it's
# the only client in this chain that can ever receive a genuine player-context PO token.
_PLAYER_CLIENT = "player_client=android_vr,web_safari,mweb"

# The bgutil-ytdlp-pot-provider plugin's own extractor-args namespace (separate from `youtube:`).
_POT_PROVIDER_EXTRACTOR_KEY = "youtubepot-bgutilhttp"

# Forces yt-dlp to attempt a PO token fetch for every context (player/gvs/subs) regardless of a
# client's static policy — see the Phase 23 module-docstring paragraph for why the default
# `fetch_pot=auto` policy never even asks for one on our client chain. Only meaningful (and only
# added) when a provider is actually configured; harmless no-op cost otherwise avoided entirely.
_FETCH_POT_ALWAYS = "fetch_pot=always"


def _extractor_args(*, extra_youtube_args: list[str] | None = None) -> list[str]:
    """Build the ``--extractor-args`` pairs shared by both yt-dlp call sites.

    Reads ``DISTIL_POT_PROVIDER_URL`` at call time (not import time) so callers — including
    tests via monkeypatch — always see the current environment. When unset, the returned args
    are byte-identical to before (``player_client`` only). When set, folds ``fetch_pot=always``
    into the *same* ``youtube:`` value (yt-dlp replaces, not merges, repeated ``--extractor-args``
    for one extractor — see Phase 23) and appends a second ``--extractor-args`` pair pointing the
    bgutil POT-provider plugin at that server's ``base_url``.

    ``extra_youtube_args`` lets a caller (currently only :func:`diagnose_pot`) fold in additional
    ``key=value`` pairs for the same ``youtube:`` namespace without stepping on
    ``player_client``/``fetch_pot`` — for the same replace-not-merge reason.
    """
    provider_url = os.environ.get("DISTIL_POT_PROVIDER_URL")
    youtube_parts = [_PLAYER_CLIENT]
    if provider_url:
        youtube_parts.append(_FETCH_POT_ALWAYS)
    if extra_youtube_args:
        youtube_parts.extend(extra_youtube_args)
    args = ["--extractor-args", f"youtube:{';'.join(youtube_parts)}"]
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
    on_phase: Callable[[str, str], None] | None = None,
) -> Transcript:
    """Fetch English captions (native ``srt``, no ffmpeg conversion needed) for one video.

    ``on_phase(phase, event)`` — if given — reports ("transcript_fetch"|"caption_parse",
    "start"|"finish") around the yt-dlp subprocess call and the srt parse respectively, purely
    for caller-side progress display; it changes nothing about what this function does.
    """
    if workdir is not None:
        # A caller-supplied workdir may be reused across fetches (e.g. tests sharing a
        # tmp_path); scope this fetch to its own unique child directory so a stale caption
        # file left behind by a previous invocation is never picked up by the glob below.
        scoped = Path(tempfile.mkdtemp(dir=str(workdir)))
        return _fetch_into(video_url, run, scoped, timeout, sleep, on_phase)
    with tempfile.TemporaryDirectory() as tmp:
        return _fetch_into(video_url, run, Path(tmp), timeout, sleep, on_phase)


def _fetch_into(
    video_url: str,
    run,
    workdir: Path,
    timeout: float,
    sleep=time.sleep,
    on_phase: Callable[[str, str], None] | None = None,
) -> Transcript:
    if on_phase is not None:
        on_phase("transcript_fetch", "start")
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
    if on_phase is not None:
        on_phase("transcript_fetch", "finish")
    srt_files = sorted(workdir.glob("*.srt"))
    if not srt_files:
        raise YoutubeFetchError("No English captions available for this video.")
    raw = srt_files[0].read_text(encoding="utf-8")
    if on_phase is not None:
        on_phase("caption_parse", "start")
    try:
        transcript = ingest_srt_text(raw)
    except IngestError as exc:
        raise YoutubeFetchError(str(exc)) from exc
    if on_phase is not None:
        on_phase("caption_parse", "finish")
    return transcript


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


# ---- PO token diagnostics (Phase 23) ----------------------------------------------------
#
# yt-dlp only prints its "PO Token Providers:" discovery line, and the bgutil plugin only logs
# its per-context "Generating a <context> PO Token for <client> client" note, at trace level
# (`pot_trace=true`) — which is also the one setting that makes the plugin log the *raw* token
# value (`Generated POT: <token>`). _redact_pot_diagnostic strips that (and the configured
# provider URL) out before any of this is ever shown to a user.

_PROVIDER_DISCOVERY_RE = re.compile(r"^.*PO Token Providers:.*$", re.MULTILINE)
_CONTEXT_ATTEMPT_RE = re.compile(r"Generating an? (?P<context>\w+) PO Token for (?P<client>\S+) client")
_POT_TOKEN_VALUE_RE = re.compile(r"(Generated POT:\s*)(\S+)")
_POT_TOKEN_REPR_RE = re.compile(r"(po_token=['\"])([^'\"]*)(['\"])")


def _redact_pot_diagnostic(text: str, provider_url: str | None) -> str:
    """Strip PO token values and the configured provider URL out of a verbose yt-dlp transcript.

    Safe to hand to a user (CLI stdout, an HTTP diagnostic response, a bug report) — unlike
    :func:`_surface_error`'s bounded failure string, this is meant to show the *whole* run, so
    what needs to stay hidden is redacted in place rather than left out by truncation.
    """
    redacted = _POT_TOKEN_VALUE_RE.sub(r"\1<redacted>", text)
    redacted = _POT_TOKEN_REPR_RE.sub(r"\1<redacted>\3", redacted)
    if provider_url:
        redacted = redacted.replace(provider_url, "<redacted-provider-url>")
    return redacted


@dataclass
class PotDiagnostic:
    """Whether a PO token was even asked for, and for which context, on one real yt-dlp run.

    ``provider_discovery`` is yt-dlp's own "[youtube] [pot] PO Token Providers: ..." line
    (``None`` if it never printed — itself diagnostic: no provider plugin registered at all).
    ``context_attempts`` lists each ``(context, client)`` pair yt-dlp actually tried to fetch a
    token for; empty means finding (a) from the task that motivated this ("never asked" — no
    attempt for *any* context), not (b) ("asked and rejected", which would show an attempt here
    followed by a provider error/rejection in ``raw_output``).
    """

    returncode: int
    provider_discovery: str | None
    context_attempts: list[tuple[str, str]] = field(default_factory=list)
    raw_output: str = ""


def diagnose_pot(
    video_url: str, *, run=subprocess.run, timeout: float = 60.0
) -> PotDiagnostic:
    """Run one verbose, no-op yt-dlp fetch for ``video_url`` and report the PO-token mechanics.

    Exists so "was a PO token even requested, and for which context" — a question that has cost
    real round trips because yt-dlp's verbose output isn't otherwise visible outside the
    container — can be answered from the running service itself (CLI: ``distil
    youtube-diagnose-pot``; web: ``GET /diagnostics/youtube-pot``), with no shell/filesystem
    access needed. Uses the exact extractor-args a real caption fetch would (:func:`_extractor_args`,
    so this reflects production behavior, not a hand-tuned command), plus ``pot_trace=true`` (the
    only way yt-dlp emits the discovery line) and ``--simulate`` (nothing written to disk). Never
    retries transient failures — a diagnostic should show exactly what happened on one real
    attempt, not silently swallow it into a different second attempt.

    Never raises: a timeout or other process-launch failure is caught here (its exception text,
    e.g. ``subprocess.TimeoutExpired``'s ``str()``, embeds the full argv including the
    ``--extractor-args`` value carrying the provider URL) and reported as a
    :class:`PotDiagnostic` with a sentinel ``returncode`` and redacted ``raw_output``, so callers
    never need their own redaction and can't reintroduce the leak by skipping it.
    """
    provider_url = os.environ.get("DISTIL_POT_PROVIDER_URL")
    try:
        proc = run(
            [
                _YT_DLP,
                "--no-update",
                "-v",
                *_extractor_args(extra_youtube_args=["pot_trace=true"]),
                "--skip-download",
                "--simulate",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en",
                "--sub-format",
                "srt/best",
                video_url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        redacted_error = _redact_pot_diagnostic(str(exc), provider_url)
        return PotDiagnostic(
            returncode=-1,
            provider_discovery=None,
            context_attempts=[],
            raw_output=redacted_error,
        )
    redacted = _redact_pot_diagnostic(f"{proc.stdout}{proc.stderr}", provider_url)
    discovery_match = _PROVIDER_DISCOVERY_RE.search(redacted)
    # A single fetch can retry the same (context, client) pair once per format candidate —
    # dedupe (preserving first-seen order) so the summary reads as "which pairs", not a wall of
    # near-identical repeats; the full detail is still in raw_output.
    attempts = list(dict.fromkeys(
        (m.group("context"), m.group("client")) for m in _CONTEXT_ATTEMPT_RE.finditer(redacted)
    ))
    return PotDiagnostic(
        returncode=proc.returncode,
        provider_discovery=discovery_match.group(0).strip() if discovery_match else None,
        context_attempts=attempts,
        raw_output=redacted,
    )
