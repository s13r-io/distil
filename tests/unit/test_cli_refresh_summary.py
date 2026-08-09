"""``distil refresh-summary <entry_id>`` — CLI wiring for the per-entry narrative-summary
refresh action. Uses the same FakeClient seam as the rest of the CLI test suite."""

import pytest
from typer.testing import CliRunner

from distil import cli
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry

runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test-model")
    return tmp_path


def _filed_entry_with_transcript(store):
    entry = KBEntry.model_validate({
        "entry_id": "e_01",
        "source": {"title": "A talk", "captured_at": "2026-01-01T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01", "type": "heuristic", "statement": "Keep functions small.",
            "stance": "opinion", "provenance": {"quote": "keep functions small"},
        }],
        "meta": {"created_at": "2026-01-01T00:00:00", "model_version": "test"},
    })
    transcript = Transcript(segments=[
        Segment(text="Keep functions small and focused on one job.", locator="seg:0"),
        Segment(text="It makes testing dramatically easier down the line.", locator="seg:1"),
    ])
    store.file_entry(entry, transcript=transcript)
    return entry


@pytest.mark.unit
def test_refresh_summary_regenerates_and_prints_confirmation(env, monkeypatch):
    store = cli._make_store()
    _filed_entry_with_transcript(store)
    monkeypatch.setattr(
        cli, "_make_summary_client", lambda: FakeClient(responses=["N" * 200])
    )

    result = runner.invoke(cli.app, ["refresh-summary", "e_01"])

    assert result.exit_code == 0
    assert "regenerated" in result.stdout.lower()
    reloaded = store.load_entry("e_01")
    assert reloaded.narrative_summary is not None
    assert reloaded.narrative_summary.text == "N" * 200


@pytest.mark.unit
def test_refresh_summary_missing_entry_is_friendly(env, monkeypatch):
    monkeypatch.setattr(cli, "_make_summary_client", lambda: FakeClient(responses=[]))
    result = runner.invoke(cli.app, ["refresh-summary", "e_missing"])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


@pytest.mark.unit
def test_refresh_summary_uses_the_summary_client_seam_not_the_main_client(env, monkeypatch):
    """The main _make_client seam must never be touched by this command — using it would be
    the strong (extraction) model, not the cheap summary tier."""
    store = cli._make_store()
    _filed_entry_with_transcript(store)

    def boom():
        raise AssertionError("refresh-summary must not construct the main client")

    monkeypatch.setattr(cli, "_make_client", boom)
    monkeypatch.setattr(
        cli, "_make_summary_client", lambda: FakeClient(responses=["N" * 200])
    )

    result = runner.invoke(cli.app, ["refresh-summary", "e_01"])
    assert result.exit_code == 0
