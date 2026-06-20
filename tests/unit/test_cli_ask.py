"""Phase 10.7 — CLI ask + reindex. Tests T-C4, T-C5."""

import json

import pytest
from typer.testing import CliRunner

from distil import cli
from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Profile

runner = CliRunner()

_TRIAGE = json.dumps({
    "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
    "density": "high", "transcript_loss": {"level": "low", "evidence": []}, "verdict": "rich",
})
_EXTRACT = json.dumps([{
    "type": "heuristic", "statement": "Write python unit tests before code.", "stance": "opinion",
    "speaker_confidence": "high",
    "provenance": {"quote": "write python unit tests", "timestamp": None, "locator": None},
}])
_LINK = json.dumps([{
    "knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
    "application_form": "checklist", "scenario": "x", "novelty_flag": False,
}])
_NOTE = json.dumps({
    "title": "Testing first",
    "core_takeaway": {"text": "Write tests before implementation code.", "item_ids": ["k_01"]},
    "key_points": [{"text": "The note is about python unit tests.", "item_ids": ["k_01"]}],
    "why_it_matters": [],
    "how_to_apply": [],
    "caveats": [],
    "review_questions": [],
    "topics": ["python testing"],
})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test-model")
    monkeypatch.setenv("DISTIL_RETRIEVAL_THRESHOLD", "0.0")
    monkeypatch.setattr(cli, "_make_embedder", lambda: FakeEmbedder(dim=64))
    store = cli._make_store()
    store.save_profile(Profile.model_validate({
        "user_id": "owner",
        "stable": {"long_term_goals": [
            {"id": "g_01", "statement": "code better", "created_at": "2026-01-01T00:00:00"}
        ]},
    }))
    return tmp_path


def _seed_entry(monkeypatch):
    monkeypatch.setattr(
        cli, "_make_client", lambda: FakeClient(responses=[_TRIAGE, _EXTRACT, _LINK, _NOTE])
    )
    runner.invoke(cli.app, ["run", "--paste", "Write python unit tests before code.", "--no-graph"])


# ---- T-C4: distil ask prints answer + sources, or the no-notes message ----


@pytest.mark.unit
def test_c4_ask_answers_with_sources(env, monkeypatch):
    _seed_entry(monkeypatch)
    synth = json.dumps({
        "answer": "Write tests first [k_01].", "cited_item_ids": ["k_01"], "conflict": None,
    })
    monkeypatch.setattr(cli, "_make_client", lambda: FakeClient(responses=[synth]))
    result = runner.invoke(cli.app, ["ask", "how should I write python"])
    assert result.exit_code == 0, result.output
    assert "Write tests first" in result.output
    assert "Sources:" in result.output
    assert "k_01" in result.output


@pytest.mark.unit
def test_c4_ask_abstains_with_no_notes(env, monkeypatch):
    _seed_entry(monkeypatch)
    monkeypatch.setenv("DISTIL_RETRIEVAL_THRESHOLD", "0.99")  # nothing clears it
    # client that would explode if called for synthesis
    monkeypatch.setattr(cli, "_make_client", lambda: FakeClient(responses=[]))
    result = runner.invoke(cli.app, ["ask", "unrelated medieval history"])
    assert result.exit_code == 0
    assert "no relevant notes" in result.output.lower()


@pytest.mark.unit
def test_c4_ask_lookup_only_lists_sources(env, monkeypatch):
    _seed_entry(monkeypatch)
    monkeypatch.setattr(cli, "_make_client", lambda: FakeClient(responses=[]))  # never called
    result = runner.invoke(cli.app, ["ask", "python", "--lookup"])
    assert result.exit_code == 0
    assert "Sources:" in result.output


# ---- T-C5: distil reindex embeds entries without a stored vector ----


@pytest.mark.unit
def test_c5_reindex_backfills(env, monkeypatch):
    # file an entry WITHOUT embedding (simulate pre-read-layer) by disabling the embedder
    monkeypatch.setattr(
        cli, "_make_client", lambda: FakeClient(responses=[_TRIAGE, _EXTRACT, _LINK, _NOTE])
    )
    monkeypatch.setattr(cli, "_safe_embedder", lambda: None)
    runner.invoke(cli.app, ["run", "--paste", "Write python unit tests before code.", "--no-graph"])
    assert cli._make_store().vector_count() == 0

    result = runner.invoke(cli.app, ["reindex"])
    assert result.exit_code == 0
    assert "Reindexed" in result.output
    assert cli._make_store().vector_count() >= 1
