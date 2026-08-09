"""External collector (Helper 2 — see AGENTS.md's "External-collector queue" entry, PR #22).

Runs on the owner's own machine, asks the server for videos it lost to YouTube's bot-identity
check, fetches them with that machine's real browser session, and hands the resulting captions
back. Pull-based, always: this module only ever *initiates* an HTTP request to the server (claim
/ submit / report-unfetchable). It never opens a listening port, accepts a callback, or is
reachable from the server in any way — the server has no address for this machine and this
module gives it none.

Shares :mod:`distil.youtube`'s fetch core (:func:`~distil.youtube.fetch_raw_captions`) rather
than reimplementing any client chain, retry/backoff, or caption handling. This module owns only
what's specific to being a *remote* collector: talking to the server's ``/collector/*`` routes
(PR #22), choosing which browser's cookies to use, pacing between fetches, and reporting the
signed-in/anonymous mode a fetch actually used — a signed-out browser still usually fetches fine
anonymously from a residential address, so the mode must be stated, never assumed.

Two retry layers, deliberately not merged:

* yt-dlp-level transient failures (429/5xx/network on the fetch itself) are already retried with
  backoff *inside* ``fetch_raw_captions`` (``distil/youtube.py``'s ``_run_yt_dlp``) — nothing to
  duplicate here.
* Collector<->server HTTP transient failures (this server briefly unreachable, a 5xx from it,
  or a response lost mid-flight) are retried here, in :meth:`CollectorClient._request`, since
  ``youtube.py`` has no reason to know this server exists. A retried ``submit_transcript`` call
  resends the exact same job_id + srt text — safe because ``/collector/jobs/{id}/transcript`` is
  idempotent by construction (PR #22): a retry landing after the first request already succeeded
  is answered with the same "queued" success, never a duplicate or an error.

A *permanent* fetch failure (a ``YoutubeFetchError`` still raised after ``fetch_raw_captions``'s
own retries are exhausted — no captions, private/deleted video, etc.) is reported through
``/collector/jobs/{id}/unfetchable`` so the job fails cleanly instead of sitting out its full
7-day ``collection_deadline`` for no reason.

The lease (``DISTIL_COLLECTOR_LEASE_SECONDS`` on the server, default 10 minutes — long enough for
one real fetch + submit) is respected structurally, not by configuration: :func:`run_collector`
always claims exactly one job at a time and finishes (submits or reports unfetchable) before
claiming another, so it never holds a video it isn't actively fetching and never claims more than
it can plausibly finish before the lease expires.

``CollectorClient``'s default opener rides out the owner's Mac's system resolver caching a bad
answer for the server's hostname right after a deploy (observed repeatedly: ``getaddrinfo`` fails
while ``dig``/``curl --resolve`` both prove the name and the server are fine) — see
:mod:`distil.collector_net` for the fallback-to-last-good-address mechanism and why it never
weakens TLS verification or outlives real DNS recovering.

Nothing here reads the knowledge base, and nothing here can act on anything the server sends
back beyond a job id and a video URL — nothing is ever executed, only fetched and returned as
text. Cookies extracted from the browser never leave this machine: only derived facts (raw
caption text, and a "signed_in"/"anonymous"/"unknown" label) are ever transmitted, logged, or
returned from this module — see ``distil.youtube._detect_browser_session``'s own docstring for
where that boundary is enforced. The collector token (``DISTIL_COLLECTOR_TOKEN``) is likewise
never logged or written anywhere by this module.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .collector_net import DNSFallbackStore, default_state_path, open_with_dns_fallback
from .youtube import YoutubeFetchError, fetch_raw_captions

_logger = logging.getLogger(__name__)

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry (2s, 4s, ...) — mirrors youtube.py


class CollectorConfigError(ValueError):
    """Raised when required collector configuration (server address, token) is missing."""


class CollectorHTTPError(Exception):
    """A collector<->server HTTP call failed after exhausting retries (if any applied).

    ``status_code`` is ``None`` for a pure transport failure (couldn't reach the server at all);
    otherwise the HTTP status the server returned. ``detail`` is a short, human-readable message
    — never the raw response body, and never anything derived from cookies or the token.
    """

    def __init__(self, status_code: int | None, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}" if status_code else detail)


@dataclass
class CollectorConfig:
    server_url: str
    token: str
    browser: str | None = None
    poll_seconds: float = 30.0
    fetch_pause_seconds: float = 5.0
    request_timeout: float = 30.0
    fetch_timeout: float = 120.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def config_from_env() -> CollectorConfig:
    """Build a :class:`CollectorConfig` from the repository's existing environment-variable
    convention (``DISTIL_*``). ``DISTIL_COLLECTOR_TOKEN`` is the same credential PR #22 already
    documents server-side; this is simply the other machine that holds it.
    """
    server_url = os.environ.get("DISTIL_COLLECTOR_SERVER_URL")
    token = os.environ.get("DISTIL_COLLECTOR_TOKEN")
    if not server_url:
        raise CollectorConfigError("DISTIL_COLLECTOR_SERVER_URL is not set.")
    if not token:
        raise CollectorConfigError("DISTIL_COLLECTOR_TOKEN is not set.")
    return CollectorConfig(
        server_url=server_url.rstrip("/"),
        token=token,
        browser=os.environ.get("DISTIL_COLLECTOR_BROWSER") or None,
        poll_seconds=_env_float("DISTIL_COLLECTOR_POLL_SECONDS", 30.0),
        fetch_pause_seconds=_env_float("DISTIL_COLLECTOR_FETCH_PAUSE_SECONDS", 5.0),
    )


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        return str(exc)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        parsed = None
    if isinstance(parsed, dict) and "detail" in parsed:
        return str(parsed["detail"])
    text = raw.decode("utf-8", errors="replace").strip()
    return text or str(exc)


class CollectorClient:
    """Thin HTTP client for the server's ``/collector/*`` routes.

    Uses :mod:`urllib.request` only — no new runtime dependency; ``distil/source.py`` already
    uses ``urllib.request.urlopen`` for the same reason. ``opener`` is injectable (default is the
    DNS-resilient opener, :func:`~distil.collector_net.open_with_dns_fallback` — see that
    module's docstring) so tests can fake the entire network boundary, mirroring
    ``distil/youtube.py``'s injectable ``run``.
    """

    def __init__(
        self,
        config: CollectorConfig,
        *,
        opener: Callable[[urllib.request.Request, float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        dns_store: DNSFallbackStore | None = None,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._dns_store = dns_store or DNSFallbackStore(default_state_path())
        self._opener = opener if opener is not None else self._resilient_default_opener

    def _resilient_default_opener(self, request: urllib.request.Request, timeout: float) -> Any:
        return open_with_dns_fallback(request, timeout, dns_store=self._dns_store)

    def claim(self, *, limit: int) -> list[dict]:
        data = self._request("POST", f"/collector/jobs/claim?limit={limit}")
        jobs = data.get("jobs")
        return jobs if isinstance(jobs, list) else []

    def submit_transcript(self, job_id: str, srt: str) -> dict:
        body = urllib.parse.urlencode({"srt": srt}).encode("utf-8")
        path = f"/collector/jobs/{urllib.parse.quote(job_id, safe='')}/transcript"
        return self._request("POST", path, body=body)

    def report_unfetchable(self, job_id: str, reason: str) -> dict:
        body = urllib.parse.urlencode({"reason": reason}).encode("utf-8")
        path = f"/collector/jobs/{urllib.parse.quote(job_id, safe='')}/unfetchable"
        return self._request("POST", path, body=body)

    def _request(self, method: str, path: str, *, body: bytes | None = None) -> dict:
        url = f"{self._config.server_url}{path}"
        headers = {"Authorization": f"Bearer {self._config.token}"}
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        last_transport_error: Exception | None = None
        for attempt in range(_HTTP_MAX_ATTEMPTS):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with self._opener(request, self._config.request_timeout) as response:
                    raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                # Only a 5xx from our own server is worth retrying — a 4xx (bad token, unknown
                # job, wrong lease state) will never succeed on retry with the same request.
                if exc.code >= 500 and attempt < _HTTP_MAX_ATTEMPTS - 1:
                    _logger.warning("Collector %s %s got HTTP %s; retrying.", method, path, exc.code)
                    self._sleep(_HTTP_RETRY_BASE_DELAY * (2**attempt))
                    continue
                raise CollectorHTTPError(exc.code, _read_error_detail(exc)) from exc
            except urllib.error.URLError as exc:
                # Covers a dropped connection / timed-out read — exactly the "lost the response"
                # case a retried submit must survive without corrupting server state (the server
                # side is idempotent for this reason; see the module docstring).
                last_transport_error = exc
                if attempt < _HTTP_MAX_ATTEMPTS - 1:
                    _logger.warning("Collector %s %s unreachable (%s); retrying.", method, path, exc)
                    self._sleep(_HTTP_RETRY_BASE_DELAY * (2**attempt))
                    continue
                raise CollectorHTTPError(None, str(exc)) from exc
        raise CollectorHTTPError(None, str(last_transport_error))


def _describe_mode(mode: str | None) -> str:
    return {
        "signed_in": "a signed-in browser session",
        "anonymous": "anonymous (no signed-in session detected in the browser's cookies)",
        "unknown": "unknown (could not determine whether the browser session was signed in)",
        None: "no browser session (DISTIL_COLLECTOR_BROWSER is not set)",
    }.get(mode, str(mode))


def _log_event(event: str, data: dict) -> None:
    """Default reporting hook: plain, human-readable lines — always states the fetch mode
    (requirement: never let an anonymous fallback pass silently). The CLI wires this in as-is;
    a caller embedding :func:`run_collector` elsewhere can pass its own ``on_event`` instead.
    """
    job_id = data.get("job_id")
    if event == "idle":
        _logger.info("No jobs waiting; sleeping.")
    elif event == "session":
        _logger.info("Job %s: fetched using %s.", job_id, _describe_mode(data.get("mode")))
    elif event == "submitted":
        _logger.info("Job %s: transcript submitted.", job_id)
    elif event == "unfetchable":
        _logger.warning("Job %s: reported unfetchable (%s).", job_id, data.get("reason"))
    elif event == "claim_failed":
        _logger.error("Could not claim work from the server: %s", data.get("detail"))
    elif event in ("submit_failed", "report_failed"):
        _logger.error("Job %s: %s failed: %s", job_id, event.removesuffix("_failed"), data.get("detail"))


def _emit(on_event: Callable[[str, dict], None] | None, event: str, data: dict) -> None:
    if on_event is not None:
        on_event(event, data)


def _process_job(
    job: dict,
    config: CollectorConfig,
    client: CollectorClient,
    fetch: Callable[..., str],
    on_event: Callable[[str, dict], None] | None,
) -> None:
    job_id = job["job_id"]
    url = job["url"]
    sessions: list[str] = []
    try:
        srt = fetch(
            url,
            cookies_from_browser=config.browser,
            on_session=sessions.append,
            timeout=config.fetch_timeout,
        )
    except YoutubeFetchError as exc:
        _emit(on_event, "session", {"job_id": job_id, "mode": sessions[0] if sessions else None})
        try:
            client.report_unfetchable(job_id, str(exc))
            _emit(on_event, "unfetchable", {"job_id": job_id, "reason": str(exc)})
        except CollectorHTTPError as report_exc:
            _emit(on_event, "report_failed", {"job_id": job_id, "detail": report_exc.detail})
        return

    mode = sessions[0] if sessions else None
    _emit(on_event, "session", {"job_id": job_id, "mode": mode})
    try:
        client.submit_transcript(job_id, srt)
        _emit(on_event, "submitted", {"job_id": job_id, "mode": mode})
    except CollectorHTTPError as exc:
        # The transcript is only ever held in this process's memory — nothing durable to retry
        # from later. The job's server-side lease will eventually expire and return it to the
        # pool for a future claim (by this collector or another) rather than corrupt anything.
        _emit(on_event, "submit_failed", {"job_id": job_id, "detail": exc.detail})


def run_collector(
    config: CollectorConfig,
    *,
    client: CollectorClient | None = None,
    fetch: Callable[..., str] = fetch_raw_captions,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
    on_event: Callable[[str, dict], None] | None = _log_event,
) -> None:
    """Pull loop: claim exactly one waiting job, fetch it, submit or report-unfetchable, pause,
    repeat. Never claims a second job while still holding/fetching one (respects the lease
    structurally — see the module docstring).

    ``max_iterations`` bounds the loop for tests; ``None`` (the CLI default) runs forever.
    """
    client = client or CollectorClient(config, sleep=sleep)
    iterations = 0
    fetched_once = False
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            jobs = client.claim(limit=1)
        except CollectorHTTPError as exc:
            _emit(on_event, "claim_failed", {"detail": exc.detail})
            sleep(config.poll_seconds)
            continue
        if not jobs:
            _emit(on_event, "idle", {})
            sleep(config.poll_seconds)
            continue
        if fetched_once:
            sleep(config.fetch_pause_seconds)
        fetched_once = True
        _process_job(jobs[0], config, client, fetch, on_event)
