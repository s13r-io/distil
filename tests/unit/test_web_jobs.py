"""WEB_UI_SPEC §8/§9/§11 — background job queue, non-blocking ingest, streaming ask.

These protect the new web surface without making real LLM calls: the worker is driven with a
fake distill_fn, and streaming is exercised via FakeClient.stream (zero network).
"""

import pytest

from distil.llm import FakeClient
from distil.query import stream_ask
from web import jobs as jobsmod

# ---- JobStore lifecycle ----------------------------------------------------------------


@pytest.fixture
def jobstore(tmp_path):
    return jobsmod.JobStore(tmp_path / "distil.db")


@pytest.mark.unit
def test_enqueue_then_claim_marks_running(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    assert job.status == jobsmod.STATUS_QUEUED
    claimed = jobstore.claim_next_queued()
    assert claimed.job_id == job.job_id
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_RUNNING


@pytest.mark.unit
def test_remove_only_legal_while_queued(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="x")
    assert jobstore.remove_queued(job.job_id) is True
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_REMOVED
    # A running job cannot be removed.
    j2 = jobstore.enqueue(kind="paste", title="t2", payload="y")
    jobstore.claim_next_queued()
    assert jobstore.remove_queued(j2.job_id) is False


@pytest.mark.unit
def test_retry_only_legal_when_failed(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="x")
    assert jobstore.retry(job.job_id) is False  # still queued
    jobstore.mark_failed(job.job_id, error="boom")
    assert jobstore.retry(job.job_id) is True
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_QUEUED
    assert jobstore.get(job.job_id).error is None


@pytest.mark.unit
def test_recover_interrupted_requeues_running(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="x")
    jobstore.claim_next_queued()  # now running
    assert jobstore.recover_interrupted() == 1
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_QUEUED


@pytest.mark.unit
def test_clear_scopes(jobstore):
    a = jobstore.enqueue(kind="paste", title="a", payload="x")
    b = jobstore.enqueue(kind="paste", title="b", payload="y")
    jobstore.mark_done(a.job_id, entry_id="e_1", summary="kept 2")
    jobstore.mark_failed(b.job_id, error="boom")
    assert jobstore.clear("finished") == 1  # only the done one
    assert jobstore.get(b.job_id).status == jobsmod.STATUS_FAILED  # failed untouched
    assert jobstore.clear("failed") == 1


# ---- Worker drives the queue with an injected distill_fn (no LLM) ----------------------


@pytest.mark.unit
def test_worker_processes_done_low_value_and_failed(tmp_path):
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    done = store.enqueue(kind="paste", title="rich", payload="x")
    low = store.enqueue(kind="paste", title="low", payload="y")
    bad = store.enqueue(kind="paste", title="bad", payload="z")

    def fake_distill(job):
        if job.title == "rich":
            return {"status": "done", "entry_id": "e_ok", "summary": "kept 3 items"}
        if job.title == "low":
            return {"status": "low_value", "entry_id": "e_lo", "summary": "nothing filed"}
        raise RuntimeError("ANTHROPIC_API_KEY missing")

    worker = jobsmod.Worker(db, fake_distill)
    assert worker.process_once() and worker.process_once() and worker.process_once()

    assert store.get(done.job_id).status == jobsmod.STATUS_DONE
    assert store.get(done.job_id).entry_id == "e_ok"
    assert store.get(low.job_id).status == jobsmod.STATUS_LOW_VALUE
    assert store.get(bad.job_id).status == jobsmod.STATUS_FAILED
    assert "ANTHROPIC_API_KEY" in store.get(bad.job_id).error


# ---- /ingest is non-blocking and /jobs reports state -----------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from distil.models import Profile
    from distil.store import Store
    from web.app import create_app

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    return TestClient(create_app())


@pytest.mark.unit
def test_ingest_paste_returns_immediately_and_queues(client):
    r = client.post("/ingest", data={"paste": "some transcript text"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    # And it shows up in the jobs list.
    jobs = client.get("/jobs", headers={"accept": "application/json"}).json()
    assert any(j["job_id"] == body["job_id"] for j in jobs)


@pytest.mark.unit
def test_ingest_empty_is_rejected(client):
    r = client.post("/ingest", data={"paste": "   "})
    assert r.status_code == 400


@pytest.mark.unit
def test_ingest_file_upload_queues(client):
    r = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello world transcript", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


@pytest.mark.unit
def test_ingest_rejects_unsupported_file(client):
    r = client.post("/ingest", files={"file": ("x.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 400


# ---- streaming ask: deltas then final; abstention makes zero synthesis calls -----------


@pytest.mark.unit
def test_stream_ask_abstains_with_no_synthesis(monkeypatch):
    """When nothing clears the threshold, stream_ask abstains and never calls the model."""
    from distil import query as q

    monkeypatch.setattr(q, "retrieve", lambda *a, **k: [])  # nothing retrieved
    client = FakeClient(responses=["should-not-be-used"])
    events = list(stream_ask("anything", store=None, embedder=None, client=client))
    assert len(events) == 1 and events[0].kind == "abstain"
    assert client.call_count == 0  # the honesty gate held


@pytest.mark.unit
def test_stream_ask_streams_then_final(monkeypatch):
    from distil import query as q

    fake_items = [
        q.RetrievedItem(item_id="k_01", entry_id="e_1", statement="s",
                        quote="qq", timestamp=None, similarity=0.9, score=0.9),
    ]
    monkeypatch.setattr(q, "retrieve", lambda *a, **k: fake_items)
    monkeypatch.setattr(q, "_detect_contradiction", lambda *a, **k: None)
    client = FakeClient(responses=["Use clear names [k_01] always."])
    events = list(stream_ask("q", store=object(), embedder=None, client=client))
    kinds = [e.kind for e in events]
    assert "delta" in kinds and kinds[-1] == "final"
    text = "".join(e.text for e in events if e.kind == "delta")
    assert "clear names" in text
    final = events[-1].result
    assert final.abstained is False
    assert "k_01" in final.cited_item_ids  # grounded citation preserved
