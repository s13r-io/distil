"""WEB_UI_SPEC §8/§9/§11 — background job queue, non-blocking ingest, streaming ask.

These protect the new web surface without making real LLM calls: the worker is driven with a
fake distill_fn, and streaming is exercised via FakeClient.stream (zero network).
"""

import json
import tempfile
from pathlib import Path

import pytest

from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.query import AskResult, ConceptRef, Source, stream_ask
from web import app as webapp
from web import jobs as jobsmod
from web.app import _ask_payload

# ---- JobStore lifecycle ----------------------------------------------------------------


@pytest.fixture
def jobstore(tmp_path):
    return jobsmod.JobStore(tmp_path / "distil.db")


@pytest.mark.unit
def test_enqueue_then_claim_marks_running(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    assert job.status == jobsmod.STATUS_QUEUED
    assert job.source_url is None
    claimed = jobstore.claim_next_queued()
    assert claimed.job_id == job.job_id
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_RUNNING


@pytest.mark.unit
def test_ask_payload_includes_source_titles_for_grouping():
    payload = _ask_payload(AskResult(
        abstained=False,
        answer="Use clear names.",
        sources=[
            Source(
                item_id="k_01",
                entry_id="e_1",
                quote="clear names",
                timestamp="00:01:00",
                entry_title="Naming Functions",
            )
        ],
    ))
    assert payload["sources"][0]["title"] == "Naming Functions"


@pytest.mark.unit
def test_ask_payload_includes_concepts_behind_the_answer():
    payload = _ask_payload(AskResult(
        abstained=False,
        answer="Use clear names.",
        concepts=[ConceptRef(concept_id="naming", title="Naming things well")],
    ))
    assert payload["concepts"] == [{"concept_id": "naming", "title": "Naming things well"}]


@pytest.mark.unit
def test_ask_payload_concepts_empty_when_none_cleared():
    payload = _ask_payload(AskResult(abstained=True, message="No relevant notes found."))
    assert payload["concepts"] == []


@pytest.mark.unit
def test_cached_embedder_reuses_loaded_instance(monkeypatch):
    calls = 0
    monkeypatch.setattr(webapp, "_EMBEDDER_CACHE", None)

    def fake_make_embedder():
        nonlocal calls
        calls += 1
        return FakeEmbedder(dim=8)

    monkeypatch.setattr(webapp, "_make_embedder", fake_make_embedder)
    first = webapp._cached_embedder()
    second = webapp._cached_embedder()
    assert first is second
    assert calls == 1


@pytest.mark.unit
def test_enqueue_persists_source_url(jobstore):
    job = jobstore.enqueue(
        kind="paste", title="t", payload="hello", source_url="https://youtu.be/abc"
    )
    assert jobstore.get(job.job_id).source_url == "https://youtu.be/abc"


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


# ---- Phase E: playlist up-front fetch — pending_fetch/fetching lifecycle -----------------


@pytest.mark.unit
def test_claim_next_pending_fetch_marks_fetching(jobstore):
    job = jobstore.enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    claimed = jobstore.claim_next_pending_fetch()
    assert claimed.job_id == job.job_id
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_FETCHING


@pytest.mark.unit
def test_mark_fetched_transitions_to_queued_with_staged_payload(jobstore):
    job = jobstore.enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    jobstore.claim_next_pending_fetch()
    jobstore.mark_fetched(job.job_id, kind=jobsmod.KIND_YOUTUBE_STAGED, payload="/staged/j1.json")
    got = jobstore.get(job.job_id)
    assert got.status == jobsmod.STATUS_QUEUED
    assert got.kind == jobsmod.KIND_YOUTUBE_STAGED
    assert got.payload == "/staged/j1.json"


@pytest.mark.unit
def test_recover_interrupted_requeues_fetching_to_pending_fetch(jobstore):
    job = jobstore.enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    jobstore.claim_next_pending_fetch()  # now fetching
    assert jobstore.recover_interrupted() == 1
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_PENDING_FETCH


@pytest.mark.unit
def test_recover_interrupted_handles_both_running_and_fetching_in_one_call(jobstore):
    running = jobstore.enqueue(kind="paste", title="a", payload="x")
    jobstore.claim_next_queued()
    fetching = jobstore.enqueue(
        kind="youtube", title="b", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    jobstore.claim_next_pending_fetch()
    assert jobstore.recover_interrupted() == 2
    assert jobstore.get(running.job_id).status == jobsmod.STATUS_QUEUED
    assert jobstore.get(fetching.job_id).status == jobsmod.STATUS_PENDING_FETCH


@pytest.mark.unit
def test_remove_allowed_while_pending_fetch_not_while_fetching(jobstore):
    job = jobstore.enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    assert jobstore.remove_queued(job.job_id) is True
    assert jobstore.get(job.job_id).status == jobsmod.STATUS_REMOVED

    j2 = jobstore.enqueue(
        kind="youtube", title="t2", payload="https://x/2", status=jobsmod.STATUS_PENDING_FETCH
    )
    jobstore.claim_next_pending_fetch()
    assert jobstore.remove_queued(j2.job_id) is False


# ---- Fetcher: batches a playlist's transcripts up front, overlapping with the distill Worker


@pytest.mark.unit
def test_fetcher_default_delay_is_three_seconds():
    fetcher = jobsmod.Fetcher("unused.db", lambda job: {"status": "fetched"})
    assert fetcher._delay_seconds == 3.0


@pytest.mark.unit
def test_fetcher_fetches_all_pending_jobs_up_front(tmp_path):
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    jobs = [
        store.enqueue(
            kind="youtube", title=f"v{i}", payload=f"https://x/{i}",
            status=jobsmod.STATUS_PENDING_FETCH,
        )
        for i in range(3)
    ]

    calls = []

    def fake_fetch(job):
        calls.append(job.payload)
        return {
            "status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED,
            "payload": f"/staged/{job.job_id}.json",
        }

    fetcher = jobsmod.Fetcher(db, fake_fetch, sleep=lambda s: None)
    while fetcher.process_once():
        pass
    assert len(calls) == 3
    for j in jobs:
        got = store.get(j.job_id)
        assert got.status == jobsmod.STATUS_QUEUED
        assert got.kind == jobsmod.KIND_YOUTUBE_STAGED


@pytest.mark.unit
def test_fetcher_pause_between_fetches_is_configurable(tmp_path):
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    for i in range(3):
        store.enqueue(
            kind="youtube", title=f"v{i}", payload=f"https://x/{i}",
            status=jobsmod.STATUS_PENDING_FETCH,
        )

    def fake_fetch(job):
        return {"status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED, "payload": "/s.json"}

    sleeps = []
    fetcher = jobsmod.Fetcher(db, fake_fetch, delay_seconds=4.5, sleep=sleeps.append)
    while fetcher.process_once():
        pass
    # 3 fetches -> a pause before the 2nd and 3rd, never before the 1st.
    assert sleeps == [4.5, 4.5]


@pytest.mark.unit
def test_fetcher_isolates_one_unfetchable_video_from_the_rest(tmp_path):
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    bad = store.enqueue(
        kind="youtube", title="no captions", payload="https://x/1",
        status=jobsmod.STATUS_PENDING_FETCH,
    )
    good = store.enqueue(
        kind="youtube", title="captioned", payload="https://x/2",
        status=jobsmod.STATUS_PENDING_FETCH,
    )

    def fake_fetch(job):
        if job.title == "no captions":
            return {"status": "failed", "error": "No English captions available for this video."}
        return {"status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED, "payload": "/s.json"}

    fetcher = jobsmod.Fetcher(db, fake_fetch, sleep=lambda s: None)
    assert fetcher.process_once() and fetcher.process_once()
    assert store.get(bad.job_id).status == jobsmod.STATUS_FAILED
    assert "captions" in store.get(bad.job_id).error
    assert store.get(good.job_id).status == jobsmod.STATUS_QUEUED


@pytest.mark.unit
def test_fetcher_isolates_unfetchable_video_when_fetch_fn_raises(tmp_path):
    """A raised exception (not just a {"status": "failed"} result) also fails only its own
    job — the same safety net Worker._process already has for distill_fn."""
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    bad = store.enqueue(
        kind="youtube", title="bad", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    good = store.enqueue(
        kind="youtube", title="good", payload="https://x/2", status=jobsmod.STATUS_PENDING_FETCH
    )

    def fake_fetch(job):
        if job.title == "bad":
            raise RuntimeError("Sign in to confirm you're not a bot")
        return {"status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED, "payload": "/s.json"}

    fetcher = jobsmod.Fetcher(db, fake_fetch, sleep=lambda s: None)
    assert fetcher.process_once() and fetcher.process_once()
    assert store.get(bad.job_id).status == jobsmod.STATUS_FAILED
    assert store.get(good.job_id).status == jobsmod.STATUS_QUEUED


@pytest.mark.unit
def test_distilling_first_video_starts_before_last_fetch_completes(tmp_path):
    """Overlap, proven deterministically: after only the first of two videos has been fetched,
    it's already distillable while the second hasn't even started fetching yet."""
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    v1 = store.enqueue(
        kind="youtube", title="v1", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH
    )
    v2 = store.enqueue(
        kind="youtube", title="v2", payload="https://x/2", status=jobsmod.STATUS_PENDING_FETCH
    )

    def fake_fetch(job):
        return {
            "status": "fetched", "kind": jobsmod.KIND_YOUTUBE_STAGED,
            "payload": f"/staged/{job.job_id}.json",
        }

    fetcher = jobsmod.Fetcher(db, fake_fetch, sleep=lambda s: None)
    assert fetcher.process_once()  # fetches v1 only

    assert store.get(v1.job_id).status == jobsmod.STATUS_QUEUED
    assert store.get(v2.job_id).status == jobsmod.STATUS_PENDING_FETCH  # untouched so far

    def fake_distill(job):
        return {"status": "done", "entry_id": "e_1", "summary": "kept 1 item"}

    worker = jobsmod.Worker(db, fake_distill)
    assert worker.process_once()  # distills v1 while v2 is still only pending_fetch
    assert store.get(v1.job_id).status == jobsmod.STATUS_DONE
    assert store.get(v2.job_id).status == jobsmod.STATUS_PENDING_FETCH  # fetch continues after

    assert fetcher.process_once()  # now fetch v2
    assert store.get(v2.job_id).status == jobsmod.STATUS_QUEUED


@pytest.mark.unit
def test_clear_scopes(jobstore):
    a = jobstore.enqueue(kind="paste", title="a", payload="x")
    b = jobstore.enqueue(kind="paste", title="b", payload="y")
    jobstore.mark_done(a.job_id, entry_id="e_1", summary="kept 2")
    jobstore.mark_failed(b.job_id, error="boom")
    assert jobstore.clear("finished") == 1  # only the done one
    assert jobstore.get(b.job_id).status == jobsmod.STATUS_FAILED  # failed untouched
    assert jobstore.clear("failed") == 1


# ---- Phase A visible progress: phases advance in order, readable from the job ----------


@pytest.mark.unit
def test_start_phase_updates_current_phase_index_total_and_readable_from_job(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    jobstore.start_phase(job.job_id, phase="ingest", index=1, total=6)
    got = jobstore.get(job.job_id)
    assert got.current_phase == "ingest"
    assert got.phase_index == 1
    assert got.phase_total == 6
    assert got.phase_started_at is not None

    jobstore.start_phase(job.job_id, phase="triage", index=2, total=6)
    got = jobstore.get(job.job_id)
    assert got.current_phase == "triage"
    assert got.phase_index == 2
    # phases advance in order — the previous phase is no longer "current".
    assert got.current_phase != "ingest"


@pytest.mark.unit
def test_record_phase_duration_persists_and_accumulates(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    jobstore.record_phase_duration(job.job_id, phase="triage", seconds=1.234)
    jobstore.record_phase_duration(job.job_id, phase="extract", seconds=5.6)
    got = jobstore.get(job.job_id)
    assert got.phase_durations == {"triage": 1.234, "extract": 5.6}
    assert got.to_dict()["phase_durations"] == {"triage": 1.234, "extract": 5.6}


@pytest.mark.unit
def test_collapse_total_reports_honestly_on_early_exit(jobstore):
    """A low-value short-circuit must shrink the declared total to what actually ran, not
    leave it claiming a total the job will never reach."""
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    jobstore.start_phase(job.job_id, phase="triage", index=3, total=9)
    jobstore.collapse_total(job.job_id)
    got = jobstore.get(job.job_id)
    assert got.phase_total == 3
    assert got.phase_index == 3


@pytest.mark.unit
def test_failed_job_retains_the_phase_it_failed_in(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    jobstore.start_phase(job.job_id, phase="extract", index=4, total=9)
    jobstore.mark_failed(job.job_id, error="boom")
    got = jobstore.get(job.job_id)
    assert got.status == jobsmod.STATUS_FAILED
    assert got.current_phase == "extract"  # names the phase it failed in


@pytest.mark.unit
def test_find_by_entry_id_returns_matching_job(jobstore):
    job = jobstore.enqueue(kind="paste", title="t", payload="hello")
    jobstore.mark_done(job.job_id, entry_id="e_1", summary="kept 1")
    found = jobstore.find_by_entry_id("e_1")
    assert found is not None and found.job_id == job.job_id
    assert jobstore.find_by_entry_id("e_missing") is None


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


@pytest.mark.unit
def test_web_distill_job_skips_inline_graph_and_reports_timings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test-model")
    monkeypatch.setattr(webapp, "_make_client", lambda: object())
    monkeypatch.setattr(webapp, "_cached_safe_embedder", lambda: None)
    monkeypatch.setattr(webapp, "_fetch_source_metadata", lambda _url: webapp.SourceMetadata())
    scheduled: list[str] = []
    monkeypatch.setattr(webapp, "_schedule_graph_link",
                        lambda entry_id: scheduled.append(entry_id) or True)
    captured: dict[str, bool] = {}

    def fake_run_pipeline(*_args, **kwargs):
        config = kwargs["config"]
        captured["enable_graph"] = config.enable_graph
        config.timing_callback("triage", 1.26)
        return KBEntry.model_validate({
            "entry_id": "e_fast",
            "source": {"title": "Fast note", "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": [{
                "item_id": "k_01",
                "type": "heuristic",
                "statement": "Keep functions small.",
                "stance": "opinion",
                "provenance": {"quote": "keep functions small"},
            }],
            "tags": {"topics": ["function_design"], "knowledge_types": ["heuristic"]},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        })

    monkeypatch.setattr(webapp, "run_pipeline", fake_run_pipeline)
    job = jobsmod.JobStore(tmp_path / "distil.db").enqueue(
        kind="paste", title="t", payload="Keep functions small."
    )
    result = webapp._distill_job(job)
    assert result["status"] == jobsmod.STATUS_DONE
    assert captured["enable_graph"] is False
    assert scheduled == ["e_fast"]
    assert "triage 1.3s" in result["summary"]
    assert "graph updating" in result["summary"]
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("distil_timing "))
    payload = json.loads(line.removeprefix("distil_timing "))
    assert payload["job_id"] == job.job_id
    assert payload["entry_id"] == "e_fast"
    assert payload["status"] == "done"
    assert payload["timings"]["triage"] == 1.26


# ---- Phase A visible progress, wired through _distill_job (no LLM) --------------------


@pytest.mark.unit
def test_distill_job_persists_phase_durations_and_current_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setattr(webapp, "_make_client", lambda: object())
    monkeypatch.setattr(webapp, "_cached_safe_embedder", lambda: None)
    monkeypatch.setattr(webapp, "_fetch_source_metadata", lambda _url: webapp.SourceMetadata())
    monkeypatch.setattr(webapp, "_schedule_graph_link", lambda entry_id: False)

    def fake_run_pipeline(*_args, **kwargs):
        config = kwargs["config"]
        # Simulate what pipeline._timed does for one stage: entry, timing, exit.
        config.phase_callback("triage", "start")
        config.timing_callback("triage", 0.5)
        config.phase_callback("triage", "finish")
        return KBEntry.model_validate({
            "entry_id": "e_fast",
            "source": {"title": "Fast note", "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
                "density": "high", "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": [{
                "item_id": "k_01", "type": "heuristic", "statement": "Keep functions small.",
                "stance": "opinion", "provenance": {"quote": "keep functions small"},
            }],
            "tags": {"topics": [], "knowledge_types": ["heuristic"]},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        })

    monkeypatch.setattr(webapp, "run_pipeline", fake_run_pipeline)
    db = tmp_path / "distil.db"
    job = jobsmod.JobStore(db).enqueue(kind="paste", title="t", payload="Keep functions small.")
    result = webapp._distill_job(job)
    assert result["status"] == jobsmod.STATUS_DONE

    got = jobsmod.JobStore(db).get(job.job_id)
    # ingest + metadata phases ran for real (not mocked) and their durations were persisted,
    # alongside the pipeline-reported "triage" duration — this is requirement 5 (durations
    # stored) plus requirement 3 (pre-pipeline steps get their own phases).
    assert got.phase_durations["triage"] >= 0
    assert "ingest" in got.phase_durations
    assert "metadata" in got.phase_durations
    assert got.current_phase == "triage"
    assert got.phase_index is not None and got.phase_total is not None
    assert got.phase_index <= got.phase_total


@pytest.mark.unit
def test_distill_job_low_value_short_circuit_collapses_total_honestly(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setattr(webapp, "_make_client", lambda: object())
    monkeypatch.setattr(webapp, "_cached_safe_embedder", lambda: None)
    monkeypatch.setattr(webapp, "_fetch_source_metadata", lambda _url: webapp.SourceMetadata())

    def fake_run_pipeline(*_args, **kwargs):
        config = kwargs["config"]
        config.phase_callback("triage", "start")
        config.timing_callback("triage", 0.2)
        config.phase_callback("triage", "finish")
        config.phase_callback("triage", "short_circuit")
        return KBEntry.model_validate({
            "entry_id": "e_low",
            "source": {"title": "vlog", "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [], "density": "low",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "little_to_extract",
            },
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        })

    monkeypatch.setattr(webapp, "run_pipeline", fake_run_pipeline)
    db = tmp_path / "distil.db"
    job = jobsmod.JobStore(db).enqueue(kind="paste", title="t", payload="meh")
    result = webapp._distill_job(job)
    assert result["status"] == jobsmod.STATUS_LOW_VALUE

    got = jobsmod.JobStore(db).get(job.job_id)
    # The declared total must shrink to what actually ran (ingest, metadata, triage) rather
    # than continuing to claim the full ~10-phase plan the run never reached.
    assert got.phase_total == got.phase_index == 3


# ---- /ingest is non-blocking and /jobs reports state -----------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from distil.models import Profile
    from distil.store import Store
    from web import app as webapp
    from web.app import create_app

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    monkeypatch.setattr(webapp, "fetch_youtube_oembed_metadata", lambda _url: webapp.SourceMetadata())
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    return TestClient(create_app())


@pytest.mark.unit
def test_ingest_paste_returns_immediately_and_queues(client):
    r = client.post(
        "/ingest",
        data={
            "paste": "some transcript text",
            "source_url": "youtube.com/watch?v=abc&feature=share&t=30s",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    # And it shows up in the jobs list.
    jobs = client.get("/jobs", headers={"accept": "application/json"}).json()
    queued = next(j for j in jobs if j["job_id"] == body["job_id"])
    assert queued["source_url"] == "https://www.youtube.com/watch?v=abc"


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


@pytest.mark.unit
def test_ingest_rejects_non_youtube_source_url(client):
    r = client.post("/ingest", data={"paste": "text", "source_url": "https://example.com"})
    assert r.status_code == 400


# ---- YouTube-only ADD input: a bare video/playlist URL (no paste, no file) --------------


@pytest.mark.unit
def test_ingest_youtube_video_url_only_queues_youtube_job(client):
    r = client.post("/ingest", data={"source_url": "https://youtu.be/abc123"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    jobs = client.get("/jobs", headers={"accept": "application/json"}).json()
    queued = next(j for j in jobs if j["job_id"] == body["job_id"])
    assert queued["source_url"] == "https://www.youtube.com/watch?v=abc123"
    assert queued["kind"] == "youtube"


@pytest.mark.unit
def test_ingest_rejects_non_youtube_url_only_source(client):
    r = client.post("/ingest", data={"source_url": "https://example.com/watch?v=abc"})
    assert r.status_code == 400


@pytest.mark.unit
def test_ingest_youtube_playlist_enqueues_one_job_per_video(client, monkeypatch):
    from distil import youtube as ytmod

    monkeypatch.setattr(
        webapp.youtube, "list_playlist_video_urls",
        lambda url, **kw: [
            "https://www.youtube.com/watch?v=vid1",
            "https://www.youtube.com/watch?v=vid2",
        ],
    )
    r = client.post(
        "/ingest", data={"source_url": "https://www.youtube.com/playlist?list=PL123"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert len(body["job_ids"]) == 2
    jobs = client.get("/jobs", headers={"accept": "application/json"}).json()
    kinds = {j["kind"] for j in jobs}
    assert kinds == {"youtube"}
    urls = {j["source_url"] for j in jobs}
    assert urls == {
        "https://www.youtube.com/watch?v=vid1", "https://www.youtube.com/watch?v=vid2",
    }
    assert ytmod is ytmod  # keep import used


@pytest.mark.unit
def test_ingest_youtube_playlist_jobs_start_pending_fetch_not_queued(client, monkeypatch):
    """A playlist video's transcript isn't fetched inline anymore — it waits for the Fetcher —
    so it must not start life already `queued` for distill (WEB_UI_SPEC honesty requirement)."""
    monkeypatch.setattr(
        webapp.youtube, "list_playlist_video_urls",
        lambda url, **kw: ["https://www.youtube.com/watch?v=vid1"],
    )
    r = client.post(
        "/ingest", data={"source_url": "https://www.youtube.com/playlist?list=PL999"}
    )
    job_id = r.json()["job_ids"][0]
    jobs = client.get("/jobs", headers={"accept": "application/json"}).json()
    job = next(j for j in jobs if j["job_id"] == job_id)
    assert job["status"] == "pending_fetch"


@pytest.mark.unit
def test_jobs_remove_cleans_up_staged_transcript_file(client, tmp_path):
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    transcript = webapp.Transcript(
        segments=[webapp.Segment(text="hi", locator="seg:0", timestamp=None)]
    )
    staged = webapp._stage_transcript("j_http_remove", transcript)
    job = store.enqueue(kind=webapp.jobsmod.KIND_YOUTUBE_STAGED, title="t", payload=str(staged))
    assert staged.exists()

    r = client.post(f"/jobs/{job.job_id}/remove")
    assert r.status_code == 200
    assert not staged.exists()


@pytest.mark.unit
def test_ingest_youtube_playlist_listing_failure_returns_400(client, monkeypatch):
    from distil.youtube import YoutubeFetchError

    def boom(url, **kw):
        raise YoutubeFetchError("Could not list playlist: playlist does not exist")

    monkeypatch.setattr(webapp.youtube, "list_playlist_video_urls", boom)
    r = client.post(
        "/ingest", data={"source_url": "https://www.youtube.com/playlist?list=bad"}
    )
    assert r.status_code == 400
    assert "playlist" in r.json()["detail"].lower()


# ---- worker: a bad video in a batch is skipped, others still process (never fatal) ------


@pytest.mark.unit
def test_load_job_transcript_dispatches_youtube_kind(monkeypatch):
    from distil.ingest import Transcript

    sentinel = Transcript(segments=[])
    captured = {}

    def fake_fetch(url, **kw):
        captured["url"] = url
        return sentinel

    monkeypatch.setattr(webapp.youtube, "fetch_video_transcript", fake_fetch)
    job = jobsmod.Job(
        job_id="j1", kind="youtube", title="t",
        payload="https://www.youtube.com/watch?v=abc", source_url=None,
        status="queued", entry_id=None, summary=None, error=None,
        created_at="", updated_at="",
    )
    result = webapp._load_job_transcript(job)
    assert result is sentinel
    assert captured["url"] == "https://www.youtube.com/watch?v=abc"


@pytest.mark.unit
def test_load_job_transcript_dispatches_youtube_staged_kind_and_deletes_staged_file(tmp_path):
    staged = tmp_path / "j1.json"
    staged.write_text(
        json.dumps({"segments": [{"text": "hi", "locator": "seg:0", "timestamp": None}]}),
        encoding="utf-8",
    )
    job = jobsmod.Job(
        job_id="j1", kind=jobsmod.KIND_YOUTUBE_STAGED, title="t", payload=str(staged),
        source_url=None, status="queued", entry_id=None, summary=None, error=None,
        created_at="", updated_at="",
    )
    result = webapp._load_job_transcript(job)
    assert result.segments[0].text == "hi"
    assert not staged.exists()  # consumed, not leaked


@pytest.mark.unit
def test_fetch_playlist_video_stages_transcript_and_reports_fetched(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    transcript = Transcript(
        segments=[Segment(text="hello", locator="seg:0", timestamp="00:00:01")]
    )
    monkeypatch.setattr(webapp.youtube, "fetch_video_transcript", lambda url, **kw: transcript)
    job = jobsmod.JobStore(tmp_path / "distil.db").enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH,
    )
    result = webapp._fetch_playlist_video(job)
    assert result["status"] == "fetched"
    assert result["kind"] == jobsmod.KIND_YOUTUBE_STAGED
    staged = Path(result["payload"])
    assert staged.exists()
    reloaded = webapp._load_staged_transcript(staged)
    assert reloaded.segments[0].text == "hello"


@pytest.mark.unit
def test_fetch_playlist_video_reports_failed_on_youtube_fetch_error(tmp_path, monkeypatch):
    from distil.youtube import YoutubeFetchError

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))

    def boom(url, **kw):
        raise YoutubeFetchError("No English captions available for this video.")

    monkeypatch.setattr(webapp.youtube, "fetch_video_transcript", boom)
    job = jobsmod.JobStore(tmp_path / "distil.db").enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH,
    )
    result = webapp._fetch_playlist_video(job)
    assert result["status"] == "failed"
    assert "captions" in result["error"]


# ---- Staging on the persistent volume (Phase E) — fixes the ephemeral-tmp upload bug ---


@pytest.mark.unit
def test_upload_dir_is_derived_from_db_path_not_ephemeral_tmp(tmp_path, monkeypatch):
    """Regression: on main, uploads are staged under tempfile.gettempdir(), which a container
    restart/redeploy wipes even though the job row (in sqlite, on the volume) survives — the job
    then fails with "file not found" and nothing to recover. Staging must live alongside the
    configured db (the persistent volume), not the ephemeral system temp dir."""
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    upload_dir = webapp._upload_dir()
    assert tmp_path == upload_dir.parent.parent
    assert not str(upload_dir).startswith(tempfile.gettempdir())


@pytest.mark.unit
def test_staging_dir_overridable_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "unused" / "distil.db"))
    monkeypatch.setenv("DISTIL_STAGING_DIR", str(tmp_path / "custom_staging"))
    assert webapp._upload_dir().parent == tmp_path / "custom_staging"


@pytest.mark.unit
def test_staged_transcript_survives_a_simulated_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    transcript = Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp=None)])
    staged_path = webapp._stage_transcript("j_restart", transcript)
    assert staged_path.exists()

    # "Restart" = nothing survives except the filesystem and sqlite file (both on the volume) —
    # simulate it with fresh objects reading the same paths, no shared in-memory state at all.
    reloaded = webapp._load_staged_transcript(staged_path)
    assert reloaded.segments[0].text == "hello"


@pytest.mark.unit
def test_staged_file_cleaned_up_when_its_job_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    transcript = Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp=None)])
    staged_path = webapp._stage_transcript("j_remove", transcript)
    job = jobsmod.Job(
        job_id="j_remove", kind=jobsmod.KIND_YOUTUBE_STAGED, title="t", payload=str(staged_path),
        source_url=None, status=jobsmod.STATUS_QUEUED, entry_id=None, summary=None, error=None,
        created_at="", updated_at="",
    )
    assert staged_path.exists()
    webapp._cleanup_staged_file(job)
    assert not staged_path.exists()


@pytest.mark.unit
def test_recovered_running_job_still_finds_its_staged_transcript(tmp_path, monkeypatch):
    """A job left `running` by a crash mid-distill is recovered to `queued` on restart — its
    staged transcript (kind youtube_staged) must still be readable, since nothing ever got the
    chance to unlink it."""
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    job = store.enqueue(
        kind="youtube", title="t", payload="https://x/1", status=jobsmod.STATUS_PENDING_FETCH,
    )
    transcript = Transcript(segments=[Segment(text="hi", locator="seg:0", timestamp=None)])
    staged_path = webapp._stage_transcript(job.job_id, transcript)
    store.mark_fetched(job.job_id, kind=jobsmod.KIND_YOUTUBE_STAGED, payload=str(staged_path))
    store.claim_next_queued()  # distill Worker claims it -> running, then the process "crashes"
    assert store.get(job.job_id).status == jobsmod.STATUS_RUNNING

    # Simulate a restart: a fresh JobStore connection, same db file + same staged file on disk.
    fresh_store = jobsmod.JobStore(db)
    assert fresh_store.recover_interrupted() == 1
    recovered = fresh_store.get(job.job_id)
    assert recovered.status == jobsmod.STATUS_QUEUED
    assert recovered.kind == jobsmod.KIND_YOUTUBE_STAGED

    transcript_again = webapp._load_job_transcript(recovered)
    assert transcript_again.segments[0].text == "hi"


@pytest.mark.unit
def test_worker_reports_uncaptioned_video_failure_without_blocking_next_job(tmp_path):
    from distil.youtube import YoutubeFetchError

    db = tmp_path / "distil.db"
    store = jobsmod.JobStore(db)
    bad = store.enqueue(kind="youtube", title="no captions", payload="https://x/1")
    good = store.enqueue(kind="youtube", title="captioned", payload="https://x/2")

    def fake_distill(job):
        if job.title == "no captions":
            raise YoutubeFetchError("No English captions available for this video.")
        return {"status": "done", "entry_id": "e_ok", "summary": "kept 1 item"}

    worker = jobsmod.Worker(db, fake_distill)
    assert worker.process_once() and worker.process_once()
    assert store.get(bad.job_id).status == jobsmod.STATUS_FAILED
    assert "captions" in store.get(bad.job_id).error
    assert store.get(good.job_id).status == jobsmod.STATUS_DONE


# ---- streaming ask: deltas then final; abstention makes zero synthesis calls -----------


@pytest.mark.unit
def test_stream_ask_abstains_with_no_synthesis(monkeypatch):
    """When nothing clears the threshold, stream_ask abstains and never calls the model."""
    from distil import query as q

    monkeypatch.setattr(q, "retrieve", lambda *a, **k: [])  # nothing retrieved
    monkeypatch.setattr(q, "retrieve_concepts", lambda *a, **k: [])  # nothing retrieved
    monkeypatch.setattr(q, "retrieve_entities", lambda *a, **k: [])  # nothing retrieved
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
    monkeypatch.setattr(q, "retrieve_concepts", lambda *a, **k: [])
    monkeypatch.setattr(q, "retrieve_entities", lambda *a, **k: [])
    monkeypatch.setattr(q, "_detect_contradiction", lambda *a, **k: None)
    client = FakeClient(responses=[
        '{"answer":"Use clear names [k_01] always.","cited_item_ids":["k_01"],"conflict":null}'
    ])
    events = list(stream_ask("q", store=object(), embedder=None, client=client))
    kinds = [e.kind for e in events]
    assert "delta" in kinds and kinds[-1] == "final"
    text = "".join(e.text for e in events if e.kind == "delta")
    assert text == "Use clear names always."
    assert "answer" not in text
    assert "k_01" not in text
    final = events[-1].result
    assert final.abstained is False
    assert "k_01" in final.cited_item_ids  # grounded citation preserved
