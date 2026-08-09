"""Helper 2 — CLI `collector-run` wiring only. distil.collector's real behavior is covered by
tests/unit/test_collector.py against a faked server; here we only check config-error handling
and that the loop function is actually invoked, monkeypatching `run_collector`/`config_from_env`
rather than touching any real network or browser."""

import pytest
from typer.testing import CliRunner

from distil import cli
from distil.collector import CollectorConfig, CollectorConfigError

runner = CliRunner()


@pytest.mark.unit
def test_collector_run_fails_cleanly_without_configuration(monkeypatch):
    def raise_missing_config():
        raise CollectorConfigError("DISTIL_COLLECTOR_SERVER_URL is not set.")

    monkeypatch.setattr(cli, "config_from_env", raise_missing_config)
    result = runner.invoke(cli.app, ["collector-run"])
    assert result.exit_code != 0
    assert "DISTIL_COLLECTOR_SERVER_URL is not set." in result.output


@pytest.mark.unit
def test_collector_run_starts_the_loop_with_env_config(monkeypatch):
    config = CollectorConfig(server_url="https://distil.example", token="secret", browser="chrome")
    calls = []
    monkeypatch.setattr(cli, "config_from_env", lambda: config)
    monkeypatch.setattr(cli, "run_collector", lambda c: calls.append(c))
    result = runner.invoke(cli.app, ["collector-run"])
    assert result.exit_code == 0
    assert calls == [config]
    assert "https://distil.example" in result.output
    assert "chrome" in result.output
    assert "secret" not in result.output  # the token itself must never be echoed


@pytest.mark.unit
def test_collector_run_reports_no_browser_configured(monkeypatch):
    config = CollectorConfig(server_url="https://distil.example", token="secret", browser=None)
    monkeypatch.setattr(cli, "config_from_env", lambda: config)
    monkeypatch.setattr(cli, "run_collector", lambda c: None)
    result = runner.invoke(cli.app, ["collector-run"])
    assert result.exit_code == 0
    assert "none configured" in result.output


@pytest.mark.unit
def test_collector_run_stops_cleanly_on_keyboard_interrupt(monkeypatch):
    config = CollectorConfig(server_url="https://distil.example", token="secret")
    monkeypatch.setattr(cli, "config_from_env", lambda: config)

    def raise_interrupt(c):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_collector", raise_interrupt)
    result = runner.invoke(cli.app, ["collector-run"])
    assert result.exit_code == 0
    assert "stopped" in result.output.lower()
