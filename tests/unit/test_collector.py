"""Helper 2 — distil/collector.py: pull-based external collector.

``CollectorClient`` talks to the server's ``/collector/*`` routes through a real FastAPI app
(``web.app.create_app()``, PR #22) via ``TestClient``, wrapped as an injectable urllib "opener"
so these tests exercise the collector's actual HTTP framing (query strings, form-encoded bodies,
Bearer header) against the real route handlers — the exact contract
``tests/unit/test_web_collector.py`` already verifies from the server side. No real network and
no real browser: ``distil.youtube``'s injected ``run`` boundary (via a fake ``fetch`` here) and
this opener fake the process/socket boundaries respectively.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request

import pytest
from fastapi.testclient import TestClient

from distil.collector import (
    CollectorClient,
    CollectorConfig,
    CollectorConfigError,
    CollectorHTTPError,
    config_from_env,
    run_collector,
)
from distil.models import Profile
from distil.store import Store
from distil.youtube import YoutubeFetchError
from web import jobs as jobsmod
from web.app import create_app

_GOOD_SRT = "1\n00:00:00,000 --> 00:00:02,000\nHello world.\n"


def _test_client_opener(test_client: TestClient):
    """Adapts a FastAPI TestClient to the ``opener(request, timeout)`` shape CollectorClient
    expects from ``urllib.request.urlopen`` — same status-raises-HTTPError, same context-manager
    ``.read()`` contract, but backed by the real app object instead of a real socket."""

    def opener(request: urllib.request.Request, timeout: float):
        parsed = urllib.parse.urlparse(request.full_url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        headers = dict(request.header_items())
        resp = test_client.request(
            request.get_method(), path, headers=headers, content=request.data
        )
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(
                path, resp.status_code, resp.reason_phrase, resp.headers,
                io.BytesIO(resp.content),
            )
        return io.BytesIO(resp.content)

    return opener


class _FlakyOpener:
    """Wraps a real opener but raises a transport error on the first N calls — simulates a
    dropped connection to the server, never a real network call."""

    def __init__(self, real_opener, fail_times: int):
        self._real_opener = real_opener
        self._fail_times = fail_times
        self.calls = 0

    def __call__(self, request, timeout):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise urllib.error.URLError("connection reset")
        return self._real_opener(request, timeout)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "collector-secret")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    return TestClient(create_app())


def _store(tmp_path) -> jobsmod.JobStore:
    return jobsmod.JobStore(tmp_path / "distil.db")


def _config(**overrides) -> CollectorConfig:
    base = {"server_url": "https://distil.example", "token": "collector-secret"}
    base.update(overrides)
    return CollectorConfig(**base)


def _client(server, **config_overrides) -> CollectorClient:
    return CollectorClient(
        _config(**config_overrides), opener=_test_client_opener(server), sleep=lambda s: None
    )


def _enqueue_waiting(store, i: int = 0):
    return store.enqueue(
        kind="youtube", title=f"t{i}", payload=f"https://youtu.be/{i}",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )


def _fake_fetch_success(url, **kwargs):
    on_session = kwargs.get("on_session")
    if on_session is not None:
        on_session("signed_in")
    return _GOOD_SRT


def _fake_fetch_permanent_failure(url, **kwargs):
    on_session = kwargs.get("on_session")
    if on_session is not None:
        on_session("anonymous")
    raise YoutubeFetchError("No English captions available for this video.")


# ---- configuration -----------------------------------------------------------------------


@pytest.mark.unit
def test_config_from_env_requires_server_url(monkeypatch):
    monkeypatch.delenv("DISTIL_COLLECTOR_SERVER_URL", raising=False)
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "x")
    with pytest.raises(CollectorConfigError):
        config_from_env()


@pytest.mark.unit
def test_config_from_env_requires_token(monkeypatch):
    monkeypatch.setenv("DISTIL_COLLECTOR_SERVER_URL", "https://distil.example")
    monkeypatch.delenv("DISTIL_COLLECTOR_TOKEN", raising=False)
    with pytest.raises(CollectorConfigError):
        config_from_env()


@pytest.mark.unit
def test_config_from_env_reads_optional_settings(monkeypatch):
    monkeypatch.setenv("DISTIL_COLLECTOR_SERVER_URL", "https://distil.example/")
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "secret")
    monkeypatch.setenv("DISTIL_COLLECTOR_BROWSER", "chrome:Default")
    monkeypatch.setenv("DISTIL_COLLECTOR_POLL_SECONDS", "45")
    monkeypatch.setenv("DISTIL_COLLECTOR_FETCH_PAUSE_SECONDS", "9")
    config = config_from_env()
    assert config.server_url == "https://distil.example"  # trailing slash stripped
    assert config.token == "secret"
    assert config.browser == "chrome:Default"
    assert config.poll_seconds == 45.0
    assert config.fetch_pause_seconds == 9.0


# ---- CollectorClient <-> real routes: claim / submit / report round trip -------------------


@pytest.mark.unit
def test_claim_returns_waiting_job(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    jobs = _client(server).claim(limit=1)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job.job_id
    assert jobs[0]["url"] == "https://youtu.be/0"
    assert store.get(job.job_id).status == jobsmod.STATUS_COLLECTING


@pytest.mark.unit
def test_claim_returns_empty_when_nothing_waiting(server):
    assert _client(server).claim(limit=1) == []


@pytest.mark.unit
def test_submit_transcript_queues_the_job(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    store.claim_for_collection(limit=5)
    result = _client(server).submit_transcript(job.job_id, _GOOD_SRT)
    assert result == {"job_id": job.job_id, "status": "queued"}
    assert store.get(job.job_id).status == jobsmod.STATUS_QUEUED


@pytest.mark.unit
def test_report_unfetchable_fails_the_job(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    store.claim_for_collection(limit=5)
    result = _client(server).report_unfetchable(job.job_id, "private video")
    assert result == {"ok": True}
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_FAILED
    assert got.error == "private video"


@pytest.mark.unit
def test_wrong_token_raises_collector_http_error(server, tmp_path):
    _enqueue_waiting(_store(tmp_path))
    client = _client(server, token="wrong-token")
    with pytest.raises(CollectorHTTPError) as excinfo:
        client.claim(limit=1)
    assert excinfo.value.status_code == 401


# ---- run_collector: full claim -> fetch -> submit loop, against the real routes ------------


@pytest.mark.unit
def test_run_collector_collects_a_blocked_video_end_to_end(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    events = []
    run_collector(
        _config(), client=_client(server), fetch=_fake_fetch_success, sleep=lambda s: None,
        max_iterations=1, on_event=lambda e, d: events.append((e, d)),
    )
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_QUEUED
    assert got.kind == jobsmod.KIND_YOUTUBE_STAGED
    assert ("session", {"job_id": job.job_id, "mode": "signed_in"}) in events
    assert ("submitted", {"job_id": job.job_id, "mode": "signed_in"}) in events
    # Proves the full chain to "distilled on the server": once submitted, the unmodified
    # distill Worker sees exactly an ordinary queued job and can claim it, indistinguishable
    # from a playlist prefetch (mirrors PR #22's own end-to-end evidence).
    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == jobsmod.STATUS_RUNNING


@pytest.mark.unit
def test_run_collector_reports_permanent_failure_as_unfetchable(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    events = []
    run_collector(
        _config(), client=_client(server), fetch=_fake_fetch_permanent_failure,
        sleep=lambda s: None, max_iterations=1, on_event=lambda e, d: events.append((e, d)),
    )
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_FAILED
    assert got.error == "No English captions available for this video."
    assert ("unfetchable", {"job_id": job.job_id, "reason": got.error}) in events


@pytest.mark.unit
def test_run_collector_idles_when_nothing_waiting(server):
    events = []
    run_collector(
        _config(), client=_client(server), sleep=lambda s: None, max_iterations=1,
        on_event=lambda e, d: events.append((e, d)),
    )
    assert events == [("idle", {})]


@pytest.mark.unit
def test_run_collector_pauses_between_fetches_but_not_before_the_first(server, tmp_path):
    store = _store(tmp_path)
    _enqueue_waiting(store, 0)
    _enqueue_waiting(store, 1)
    sleeps = []
    config = _config(fetch_pause_seconds=7.0)
    client = CollectorClient(config, opener=_test_client_opener(server), sleep=lambda s: None)
    run_collector(
        config, client=client, fetch=_fake_fetch_success, sleep=sleeps.append,
        max_iterations=2, on_event=None,
    )
    # Two jobs fetched across two iterations: paused once, between them - never before the first.
    assert sleeps == [7.0]


@pytest.mark.unit
def test_run_collector_polls_at_configured_interval_when_idle(server):
    sleeps = []
    config = _config(poll_seconds=11.0)
    client = CollectorClient(config, opener=_test_client_opener(server), sleep=lambda s: None)
    run_collector(config, client=client, sleep=sleeps.append, max_iterations=1, on_event=None)
    assert sleeps == [11.0]


@pytest.mark.unit
def test_run_collector_never_claims_more_than_one_job_at_once(server, tmp_path):
    """Respect-the-lease: only the job actively being fetched is ever leased; the rest stay in
    the waiting pool rather than being claimed-but-idle."""
    store = _store(tmp_path)
    for i in range(3):
        _enqueue_waiting(store, i)
    run_collector(
        _config(), client=_client(server), fetch=_fake_fetch_success, sleep=lambda s: None,
        max_iterations=1, on_event=None,
    )
    active = store.list_active()
    waiting = [j for j in active if j.status == jobsmod.STATUS_AWAITING_COLLECTION]
    collecting = [j for j in active if j.status == jobsmod.STATUS_COLLECTING]
    assert len(waiting) == 2
    assert collecting == []


# ---- signed-in vs anonymous detection is reported, per fetch -------------------------------


@pytest.mark.unit
def test_signed_in_vs_anonymous_mode_is_detected_and_reported(server, tmp_path):
    store = _store(tmp_path)
    job = _enqueue_waiting(store)

    def fake_fetch(url, **kwargs):
        kwargs["on_session"]("anonymous")
        return _GOOD_SRT

    events = []
    run_collector(
        _config(), client=_client(server), fetch=fake_fetch, sleep=lambda s: None,
        max_iterations=1, on_event=lambda e, d: events.append((e, d)),
    )
    assert ("session", {"job_id": job.job_id, "mode": "anonymous"}) in events


@pytest.mark.unit
def test_no_browser_configured_is_reported_as_its_own_mode(server, tmp_path):
    """DISTIL_COLLECTOR_BROWSER unset -> fetch is called with cookies_from_browser=None, and
    that must be visible as its own mode, never conflated with a detected anonymous session."""
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    seen_cookies_from_browser = []

    def fake_fetch(url, **kwargs):
        seen_cookies_from_browser.append(kwargs.get("cookies_from_browser"))
        # No browser configured -> fetch_raw_captions would never call on_session at all.
        return _GOOD_SRT

    events = []
    run_collector(
        _config(), client=_client(server), fetch=fake_fetch, sleep=lambda s: None,
        max_iterations=1, on_event=lambda e, d: events.append((e, d)),
    )
    assert seen_cookies_from_browser == [None]
    assert ("session", {"job_id": job.job_id, "mode": None}) in events


# ---- transient HTTP failures: retry with backoff, without corrupting server state ----------


@pytest.mark.unit
def test_transient_transport_failure_retries_with_backoff_then_succeeds(server, tmp_path):
    _enqueue_waiting(_store(tmp_path))
    flaky = _FlakyOpener(_test_client_opener(server), fail_times=1)
    sleeps = []
    client = CollectorClient(_config(), opener=flaky, sleep=sleeps.append)
    jobs = client.claim(limit=1)
    assert len(jobs) == 1
    assert flaky.calls == 2
    assert sleeps == [2.0]


@pytest.mark.unit
def test_transient_5xx_from_server_retries_with_backoff_then_succeeds(server, tmp_path):
    _enqueue_waiting(_store(tmp_path))
    real_opener = _test_client_opener(server)
    calls = {"n": 0}

    def opener(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", None, io.BytesIO(b"{}"),
            )
        return real_opener(request, timeout)

    sleeps = []
    client = CollectorClient(_config(), opener=opener, sleep=sleeps.append)
    jobs = client.claim(limit=1)
    assert len(jobs) == 1
    assert calls["n"] == 2
    assert sleeps == [2.0]


@pytest.mark.unit
def test_persistent_transport_failure_raises_after_bounded_retries(server):
    flaky = _FlakyOpener(_test_client_opener(server), fail_times=99)
    sleeps = []
    client = CollectorClient(_config(), opener=flaky, sleep=sleeps.append)
    with pytest.raises(CollectorHTTPError):
        client.claim(limit=1)
    assert flaky.calls == 3  # bounded — not retried forever
    assert sleeps == [2.0, 4.0]


@pytest.mark.unit
def test_a_4xx_from_the_server_is_never_retried(server):
    """A wrong token (401) or a bad job id (404) will never succeed on retry — retrying it
    would just be noise against the server, and constraint (5) asks for good behavior there."""
    calls = {"n": 0}
    real_opener = _test_client_opener(server)

    def opener(request, timeout):
        calls["n"] += 1
        return real_opener(request, timeout)

    client = CollectorClient(_config(token="wrong-token"), opener=opener, sleep=lambda s: None)
    with pytest.raises(CollectorHTTPError) as excinfo:
        client.claim(limit=1)
    assert excinfo.value.status_code == 401
    assert calls["n"] == 1


@pytest.mark.unit
def test_a_lost_submit_response_is_retried_and_does_not_double_submit(server, tmp_path):
    """The core idempotency proof: a transport error the collector itself perceives while
    reading the response (the server already processed the request) triggers a retry of the
    exact same submit — the server's idempotent submit_collected_transcript (PR #22) must
    answer success again, never a duplicate or corrupted state."""
    store = _store(tmp_path)
    job = _enqueue_waiting(store)
    store.claim_for_collection(limit=5)

    real_opener = _test_client_opener(server)
    calls = {"n": 0}

    def opener_that_loses_the_first_response(request, timeout):
        calls["n"] += 1
        response = real_opener(request, timeout)  # server-side effect already happened
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset while reading response")
        return response

    client = CollectorClient(
        _config(), opener=opener_that_loses_the_first_response, sleep=lambda s: None
    )
    result = client.submit_transcript(job.job_id, _GOOD_SRT)
    assert result == {"job_id": job.job_id, "status": "queued"}
    assert calls["n"] == 2
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_QUEUED
    assert got.kind == jobsmod.KIND_YOUTUBE_STAGED


@pytest.mark.unit
def test_run_collector_survives_a_submit_failure_without_crashing(server, tmp_path):
    """If the server is genuinely unreachable for the whole submit (all retries exhausted),
    the loop must report and move on — never crash, never re-fetch, never corrupt state."""
    store = _store(tmp_path)
    job = _enqueue_waiting(store)

    def always_fails(request, timeout):
        raise urllib.error.URLError("server is down")

    # claim/submit both go through the same client; use the real opener for claim, then a
    # client whose opener fails only for the submit path.
    real_opener = _test_client_opener(server)

    def opener(request, timeout):
        if "/transcript" in request.full_url:
            return always_fails(request, timeout)
        return real_opener(request, timeout)

    client = CollectorClient(_config(), opener=opener, sleep=lambda s: None)
    events = []
    run_collector(
        _config(), client=client, fetch=_fake_fetch_success, sleep=lambda s: None,
        max_iterations=1, on_event=lambda e, d: events.append((e, d)),
    )
    assert any(e[0] == "submit_failed" for e in events)
    # The job is left leased (COLLECTING) rather than corrupted — it will return to the pool
    # once its lease expires, exactly like a collector that crashed mid-fetch.
    assert store.get(job.job_id).status == jobsmod.STATUS_COLLECTING
