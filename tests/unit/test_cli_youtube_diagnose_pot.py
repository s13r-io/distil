"""Phase 23 — CLI `youtube-diagnose-pot`: the permanent PO-token diagnostic capability.

`distil.youtube.diagnose_pot` itself is unit-tested against an injected `run` callable in
tests/unit/test_youtube.py; here we only cover CLI wiring (URL validation, exit formatting),
so `cli.diagnose_pot` is monkeypatched rather than touching the real subprocess boundary.
"""

import pytest
from typer.testing import CliRunner

from distil import cli
from distil.youtube import PotDiagnostic

runner = CliRunner()


@pytest.mark.unit
def test_youtube_diagnose_pot_rejects_non_youtube_url():
    result = runner.invoke(cli.app, ["youtube-diagnose-pot", "https://example.com/video"])
    assert result.exit_code != 0
    assert "YouTube URL" in result.output


@pytest.mark.unit
def test_youtube_diagnose_pot_reports_no_attempts(monkeypatch):
    monkeypatch.setattr(
        cli,
        "diagnose_pot",
        lambda url, **kwargs: PotDiagnostic(
            returncode=1, provider_discovery=None, context_attempts=[], raw_output="ERROR: bot check"
        ),
    )
    result = runner.invoke(
        cli.app, ["youtube-diagnose-pot", "https://www.youtube.com/watch?v=abc"]
    )
    assert result.exit_code == 0
    assert "none — no provider registered" in result.output
    assert "Context attempts: none" in result.output
    assert "ERROR: bot check" in result.output


@pytest.mark.unit
def test_youtube_diagnose_pot_reports_context_attempts(monkeypatch):
    monkeypatch.setattr(
        cli,
        "diagnose_pot",
        lambda url, **kwargs: PotDiagnostic(
            returncode=0,
            provider_discovery="[youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external)",
            context_attempts=[("player", "mweb"), ("gvs", "web_safari")],
            raw_output="...",
        ),
    )
    result = runner.invoke(
        cli.app, ["youtube-diagnose-pot", "https://www.youtube.com/watch?v=abc"]
    )
    assert result.exit_code == 0
    assert "bgutil:http-1.3.1" in result.output
    assert "player PO token requested for mweb client" in result.output
    assert "gvs PO token requested for web_safari client" in result.output
