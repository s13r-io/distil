"""Per-stage model resolution (distil/model_config.py) — the general mechanism that replaces
the one-off DISTIL_SUMMARY_MODEL global. Every stage resolves its own model by name."""

import pytest

from distil.model_config import (
    CHEAP_TIER_STAGES,
    DEFAULT_SUMMARY_MODEL,
    STRONG_TIER_STAGES,
    make_stage_client,
    resolve_stage_model,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DISTIL_MODEL", raising=False)
    for stage in (*STRONG_TIER_STAGES, *CHEAP_TIER_STAGES):
        monkeypatch.delenv(f"DISTIL_MODEL_{stage.upper()}", raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("stage", STRONG_TIER_STAGES)
def test_strong_stage_falls_back_to_distil_model(monkeypatch, stage):
    monkeypatch.setenv("DISTIL_MODEL", "claude-opus-5")
    assert resolve_stage_model(stage) == "claude-opus-5"


@pytest.mark.unit
@pytest.mark.parametrize("stage", STRONG_TIER_STAGES)
def test_strong_stage_raises_without_distil_model(stage):
    with pytest.raises(RuntimeError):
        resolve_stage_model(stage)


@pytest.mark.unit
@pytest.mark.parametrize("stage", STRONG_TIER_STAGES)
def test_strong_stage_per_stage_override_wins_and_is_isolated(monkeypatch, stage):
    """Overriding one stage must not leak into the others — that's the whole point of a
    per-stage mechanism rather than a second global variable."""
    monkeypatch.setenv("DISTIL_MODEL", "claude-opus-5")
    monkeypatch.setenv(f"DISTIL_MODEL_{stage.upper()}", "claude-sonnet-5")
    assert resolve_stage_model(stage) == "claude-sonnet-5"
    for other in STRONG_TIER_STAGES:
        if other != stage:
            assert resolve_stage_model(other) == "claude-opus-5"


@pytest.mark.unit
def test_summary_defaults_to_the_cheap_tier_without_any_env_var():
    assert resolve_stage_model("summary") == DEFAULT_SUMMARY_MODEL


@pytest.mark.unit
def test_summary_override_wins_over_the_cheap_default(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL_SUMMARY", "claude-opus-5")
    assert resolve_stage_model("summary") == "claude-opus-5"


@pytest.mark.unit
def test_summary_is_unaffected_by_distil_model(monkeypatch):
    """The cheap tier must never accidentally inherit the strong global."""
    monkeypatch.setenv("DISTIL_MODEL", "claude-opus-5")
    assert resolve_stage_model("summary") == DEFAULT_SUMMARY_MODEL


@pytest.mark.unit
def test_make_stage_client_uses_the_resolved_model(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    client = make_stage_client("extract")
    assert client.model == "claude-sonnet-5"


@pytest.mark.unit
def test_make_stage_client_never_requires_an_api_key_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Must not raise: the credential requirement is enforced lazily, at the first real call,
    # exactly like AnthropicClient() already does today.
    make_stage_client("summary")
