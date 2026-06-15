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

runner = CliRunner()


_TRIAGE_RICH = json.dumps({
    "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
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
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK])
    src = tmp_path / "t.txt"
    src.write_text("Keep functions small and focused.")
    result = runner.invoke(cli.app, ["run", str(src), "--no-graph"])
    assert result.exit_code == 0, result.output
    assert ".md" in result.output


@pytest.mark.unit
def test_c1_run_paste_via_option(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK])
    result = runner.invoke(
        cli.app, ["run", "--paste", "Keep functions small and focused.", "--no-graph"]
    )
    assert result.exit_code == 0, result.output
    assert ".md" in result.output


# ---- T-C2: distil score mutates the profile ----


@pytest.mark.unit
def test_c2_score_mutates_profile(env, monkeypatch):
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK])
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
    _fake(monkeypatch, [_TRIAGE_RICH, _EXTRACT, _LINK])
    run = runner.invoke(cli.app, ["run", "--paste", "Keep functions small.", "--no-graph"])
    entry_id = run.output.strip().split("/")[-1].replace(".md", "").strip()

    lst = runner.invoke(cli.app, ["list"])
    assert lst.exit_code == 0
    assert entry_id in lst.output

    show = runner.invoke(cli.app, ["show", entry_id])
    assert show.exit_code == 0
    assert "Keep functions small." in show.output


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
