"""External-collector queue (WEB_UI_SPEC collector phase): a scoped, revocable credential lets a
collector running elsewhere claim bot-checked videos, submit a transcript, or report one as
genuinely unfetchable — and reach nothing else on the site (web/auth.py's
``request_is_collector_authorized`` is a separate check from the owner's session/bearer).

No real network or yt-dlp calls here: transcripts are plain SRT text posted directly to the
route, exercising ``distil.ingest.ingest_srt_text`` the same way a real collected caption file
would.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from distil.models import Profile
from distil.store import Store
from web import jobs as jobsmod
from web.app import create_app

_VALID_SRT = "1\n00:00:00,000 --> 00:00:02,000\nHello world.\n"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
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


# ---- credential scoping: collector token can't reach anything but claim/submit -------------


@pytest.mark.unit
def test_collector_route_requires_a_token_at_all(client):
    r = client.post("/collector/jobs/claim")
    assert r.status_code == 401


@pytest.mark.unit
def test_collector_route_rejects_the_wrong_token(client):
    r = client.post("/collector/jobs/claim", headers=_bearer("wrong-token"))
    assert r.status_code == 401


@pytest.mark.unit
def test_collector_routes_unusable_when_no_token_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    monkeypatch.delenv("DISTIL_COLLECTOR_TOKEN", raising=False)
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    c = TestClient(create_app())
    r = c.post("/collector/jobs/claim", headers=_bearer("anything"))
    assert r.status_code == 401


@pytest.mark.unit
def test_collector_token_cannot_log_into_the_site_in_public_mode(tmp_path, monkeypatch):
    """The core credential-separation guarantee: a leaked collector token must not be usable as
    the owner's DISTIL_AUTH_SECRET, even though both are Bearer tokens on the same app."""
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "owner-secret")
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "collector-secret")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    c = TestClient(create_app())
    r = c.get("/entries", headers=_bearer("collector-secret"))
    assert r.status_code == 401


@pytest.mark.unit
def test_owner_token_cannot_reach_collector_routes(tmp_path, monkeypatch):
    """The reverse guarantee: the owner's own site secret does not unlock the collector queue —
    it's a genuinely separate credential, not just a differently-worded check on the same one."""
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "owner-secret")
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "collector-secret")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    c = TestClient(create_app())
    r = c.post("/collector/jobs/claim", headers=_bearer("owner-secret"))
    assert r.status_code == 401


@pytest.mark.unit
def test_collector_token_reaches_no_other_protected_route(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "owner-secret")
    monkeypatch.setenv("DISTIL_COLLECTOR_TOKEN", "collector-secret")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    c = TestClient(create_app())
    for path in ("/entries", "/library", "/ask?q=hi", "/bundle.zip"):
        r = c.get(path, headers=_bearer("collector-secret"))
        assert r.status_code == 401, path


# ---- claim ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_collector_claim_returns_waiting_jobs(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    r = client.post("/collector/jobs/claim", headers=_bearer("collector-secret"))
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job.job_id
    assert jobs[0]["url"] == "https://youtu.be/abc"
    assert jobs[0]["lease_expires_at"] is not None
    assert store.get(job.job_id).status == jobsmod.STATUS_COLLECTING


@pytest.mark.unit
def test_collector_claim_ignores_jobs_in_other_states(client, tmp_path):
    _store(tmp_path).enqueue(kind="paste", title="t", payload="hello")
    r = client.post("/collector/jobs/claim", headers=_bearer("collector-secret"))
    assert r.json()["jobs"] == []


@pytest.mark.unit
def test_collector_claim_never_returns_the_same_job_twice(client, tmp_path):
    _store(tmp_path).enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    first = client.post("/collector/jobs/claim", headers=_bearer("collector-secret")).json()
    second = client.post("/collector/jobs/claim", headers=_bearer("collector-secret")).json()
    assert len(first["jobs"]) == 1
    assert second["jobs"] == []


# ---- submit transcript ------------------------------------------------------------------------


@pytest.mark.unit
def test_collector_submit_validates_and_queues_a_good_transcript(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    store.claim_for_collection(limit=5)
    r = client.post(
        f"/collector/jobs/{job.job_id}/transcript",
        data={"srt": _VALID_SRT},
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 200
    assert r.json() == {"job_id": job.job_id, "status": "queued"}
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_QUEUED
    assert got.kind == jobsmod.KIND_YOUTUBE_STAGED
    assert Path(got.payload).exists()


@pytest.mark.unit
def test_collector_submit_rejects_malformed_transcript_and_leaves_the_lease_intact(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    store.claim_for_collection(limit=5)
    r = client.post(
        f"/collector/jobs/{job.job_id}/transcript",
        data={"srt": "   \n\n   "},  # no subtitle cues — IngestError, not accepted
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 400
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_COLLECTING  # unchanged — never accepted into the pipeline
    assert got.kind == "youtube"


@pytest.mark.unit
def test_collector_submit_rejects_when_job_was_never_leased(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )  # waiting, but nobody claimed it
    r = client.post(
        f"/collector/jobs/{job.job_id}/transcript",
        data={"srt": _VALID_SRT},
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 409


@pytest.mark.unit
def test_collector_submit_unknown_job_404s(client):
    r = client.post(
        "/collector/jobs/j_missing/transcript",
        data={"srt": _VALID_SRT},
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 404


@pytest.mark.unit
def test_collector_submit_is_idempotent_on_retry(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    store.claim_for_collection(limit=5)
    first = client.post(
        f"/collector/jobs/{job.job_id}/transcript",
        data={"srt": _VALID_SRT},
        headers=_bearer("collector-secret"),
    )
    assert first.status_code == 200
    second = client.post(
        f"/collector/jobs/{job.job_id}/transcript",
        data={"srt": _VALID_SRT},
        headers=_bearer("collector-secret"),
    )
    assert second.status_code == 200  # a lost-response retry must not be an error
    assert store.get(job.job_id).status == jobsmod.STATUS_QUEUED  # still exactly one queued job


# ---- report unfetchable -----------------------------------------------------------------------


@pytest.mark.unit
def test_collector_report_unfetchable_marks_the_job_failed(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://youtu.be/abc",
        status=jobsmod.STATUS_AWAITING_COLLECTION,
    )
    store.claim_for_collection(limit=5)
    r = client.post(
        f"/collector/jobs/{job.job_id}/unfetchable",
        data={"reason": "private video"},
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 200
    got = store.get(job.job_id)
    assert got.status == jobsmod.STATUS_FAILED
    assert got.error == "private video"


@pytest.mark.unit
def test_collector_report_unfetchable_rejects_when_not_leased(client, tmp_path):
    store = _store(tmp_path)
    job = store.enqueue(kind="youtube", title="t", payload="https://youtu.be/abc")  # never parked
    r = client.post(
        f"/collector/jobs/{job.job_id}/unfetchable",
        data={"reason": "private video"},
        headers=_bearer("collector-secret"),
    )
    assert r.status_code == 409
