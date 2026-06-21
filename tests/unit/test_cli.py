"""Phase 9 — cli.py (Typer). Tests T-C1, T-C2, T-C3.

The CLI builds its store from env (DISTIL_DB_PATH / DISTIL_KB_DIR) and its LLM client from
``distil.cli._make_client`` — a seam tests monkeypatch with a FakeClient so no network is hit.
"""

import json

import pytest
from typer.testing import CliRunner

from distil import cli
from distil.llm import FakeClient
from distil.models import Profile
from distil.source import SourceMetadata

runner = CliRunner()


_TRIAGE_RICH = json.dumps({
    "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})
_TRIAGE_LOW = json.dumps({
    "knowledge_types_present": [],
    "density": "low", "transcript_loss": {"level": "low", "evidence": []},
    "verdict": "little_to_extract",
})
_EXTRACT = json.dumps([{
    "type": "heuristic", "statement": "Keep functions small.", "stance": "opinion",
    "speaker_confidence": "high",
    "provenance": {"quote": "keep functions small", "timestamp": None, "locator": None},
}])
_LINK = json.dumps([{
    "knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
    "application_form": "checklist", "scenario": "refactor", "novelty_flag": False,
}])
_NOTE = json.dumps({
    "title": "Small functions",
    "core_takeaway": {"text": "Small functions are easier to reason about.", "item_ids": ["k_01"]},
    "key_points": [],
    "why_it_matters": [],
    "how_to_apply": [],
    "caveats": [],
    "review_questions": [],
    "topics": ["function design"],
})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test-model")
    # seed a profile with a goal so links attach
    store = cli._make_store()
    store.save_profile(Profile.model_validate({
        "user_id": "owner",
        "stable": {"long_term_goals": [
            {"id": "g_01", "statement": "code better", "created_at": "2026-01-01T00:00:00"}
        ]},
    }))
    return tmp_path


def _fake(monkeypatch, responses):
    monkeypatch.setattr(cli, "_make_client", lambda: FakeClient(responses=responses))


# ---- T-C1: distil run <file> and --paste ----


@pytest.mark.unit
def test_c1_run_file_exits_zero_and_prints_path(env, monkeypatch, tmp_path):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    monkeypatch.setattr(
        cli,
        "fetch_youtube_oembed_metadata",
        lambda _url: SourceMetadata(
            title="Fetched Video Title",
            channel="Fetched Channel",
            channel_url="https://www.youtube.com/@fetched",
            thumbnail_url="https://i.ytimg.com/vi/abc/hqdefault.jpg",
            metadata_provider="youtube_oembed",
            metadata_fetched_at="2026-06-21T00:00:00+00:00",
        ),
    )
    src = tmp_path / "[English] my-video_title (Transcript).txt"
    src.write_text("Keep functions small and focused.")
    result = runner.invoke(
        cli.app,
        [
            "run", str(src), "--url",
            "youtube.com/watch?v=abc&feature=share&t=30s&utm_source=copy",
            "--no-graph",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ".md" in result.output
    entry_id = result.output.strip().split("/")[-1].replace(".md", "").strip()
    entry = cli._make_store().load_entry(entry_id)
    assert entry.source.title == "Fetched Video Title"
    assert entry.source.url == "https://www.youtube.com/watch?v=abc"
    assert entry.source.channel == "Fetched Channel"
    assert entry.source.thumbnail_url == "https://i.ytimg.com/vi/abc/hqdefault.jpg"


@pytest.mark.unit
def test_c1_run_paste_via_option(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    result = runner.invoke(
        cli.app, ["run", "--paste", "Keep functions small and focused.", "--no-graph"]
    )
    assert result.exit_code == 0, result.output
    assert ".md" in result.output


@pytest.mark.unit
def test_c1_low_value_run_prints_nothing_filed(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_LOW])
    result = runner.invoke(
        cli.app, ["run", "--paste", "hey guys smash that like button", "--no-graph"]
    )
    assert result.exit_code == 0, result.output
    assert "Nothing filed" in result.output
    assert cli._make_store().list_entries() == []


# ---- T-C2: distil score mutates the profile ----


@pytest.mark.unit
def test_c2_score_mutates_profile(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    run = runner.invoke(cli.app, ["run", "--paste", "Keep functions small.", "--no-graph"])
    entry_id = run.output.strip().split("/")[-1].replace(".md", "").strip()

    before = cli._make_store().load_profile("owner")
    result = runner.invoke(cli.app, ["score", entry_id, "--score", "5", "--reason", "relevant"])
    assert result.exit_code == 0, result.output
    after = cli._make_store().load_profile("owner")
    assert after.meta.documents_processed == before.meta.documents_processed + 1


# ---- T-C3: list/show + friendly errors ----


@pytest.mark.unit
def test_c3_list_and_show(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK, _NOTE])
    run = runner.invoke(cli.app, ["run", "--paste", "Keep functions small.", "--no-graph"])
    entry_id = run.output.strip().split("/")[-1].replace(".md", "").strip()

    lst = runner.invoke(cli.app, ["list"])
    assert lst.exit_code == 0
    assert entry_id in lst.output

    show = runner.invoke(cli.app, ["show", entry_id])
    assert show.exit_code == 0
    assert "Keep functions small." in show.output
    assert "Core takeaway" in show.output

    delete = runner.invoke(cli.app, ["delete", entry_id, "--yes"])
    assert delete.exit_code == 0, delete.output
    assert "Deleted" in delete.output
    assert not cli._make_store().entry_path(entry_id).exists()


@pytest.mark.unit
def test_c3_show_missing_entry_is_friendly(env, monkeypatch):
    result = runner.invoke(cli.app, ["show", "e_nope"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


@pytest.mark.unit
def test_c3_missing_api_key_is_friendly(env, monkeypatch):
    # Use the real client path (no fake) with no key set → friendly message, not a stack trace.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["run", "--paste", "some content", "--no-graph"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output
    assert "Traceback" not in result.output


@pytest.mark.unit
def test_c3_run_bad_file_is_friendly(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH])
    result = runner.invoke(cli.app, ["run", "/no/such/file.txt", "--no-graph"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


@pytest.mark.unit
def test_c3_run_bad_url_is_friendly(env, monkeypatch):
    result = runner.invoke(
        cli.app, ["run", "--paste", "some content", "--url", "https://example.com/x"]
    )
    assert result.exit_code != 0
    assert "YouTube URL" in result.output
    assert "Traceback" not in result.output
