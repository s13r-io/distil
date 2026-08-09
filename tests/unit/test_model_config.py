"""Per-stage model resolution (distil/model_config.py) — the general mechanism that replaces
the one-off DISTIL_SUMMARY_MODEL global. Every stage resolves its own model by name."""

import pytest

from distil.model_config import (
    CHEAP_TIER_STAGES,
    DEFAULT_CHEAP_TIER_MODEL,
    DEFAULT_SUMMARY_MODEL,
    MODEL_MAX_OUTPUT_TOKENS,
    STRONG_TIER_STAGES,
    make_stage_client,
    resolve_stage_max_tokens,
    resolve_stage_model,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTIL_MODEL", raising=False)
    for stage in (*STRONG_TIER_STAGES, *CHEAP_TIER_STAGES):
        monkeypatch.delenv(f"DISTIL_MODEL_{stage.upper()}", raising=False)
        monkeypatch.delenv(f"DISTIL_MAX_TOKENS_{stage.upper()}", raising=False)
    # Point at a database file that never exists in this test run, so resolve_stage_model's
    # stored-setting lookup (distil/model_settings.py) never finds a row left over from a real
    # ./data/distil.db — proves this suite's env/default-only behavior is unaffected by the
    # settings surface unless a test explicitly stores something.
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))


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


# ---- Per-stage max_tokens ceiling (T-MCFG5) --------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("model,expected", list(MODEL_MAX_OUTPUT_TOKENS.items()))
def test_extract_resolves_to_the_models_published_output_ceiling(monkeypatch, model, expected):
    """Extraction is the one stage whose ceiling scales with the resolved model's real output
    limit — sourced from Anthropic's own model catalog (see model_config.py's module comment
    for where), not guessed as 4096/8192/some other round number."""
    monkeypatch.setenv("DISTIL_MODEL", model)
    assert resolve_stage_max_tokens("extract") == expected


@pytest.mark.unit
@pytest.mark.parametrize("stage", [s for s in STRONG_TIER_STAGES if s != "extract"])
def test_non_extract_strong_stage_keeps_the_default_ceiling(monkeypatch, stage):
    """A small classification/synthesis response doesn't need extraction's headroom — every
    stage but extract keeps the pre-existing flat default regardless of which model is
    configured."""
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    assert resolve_stage_max_tokens(stage) == 4096


@pytest.mark.unit
def test_summary_keeps_the_default_ceiling():
    assert resolve_stage_max_tokens("summary") == 4096


@pytest.mark.unit
def test_extract_max_tokens_falls_back_to_default_for_an_unrecognized_model(monkeypatch):
    """An unrecognized model string must not guess a larger number — fall back to the same
    conservative default every other stage already uses."""
    monkeypatch.setenv("DISTIL_MODEL", "some-future-model-not-in-the-table")
    assert resolve_stage_max_tokens("extract") == 4096


@pytest.mark.unit
def test_max_tokens_env_override_wins_and_is_isolated(monkeypatch):
    """DISTIL_MAX_TOKENS_<STAGE> overrides one stage only — mirrors DISTIL_MODEL_<STAGE>'s
    isolation guarantee (test_strong_stage_per_stage_override_wins_and_is_isolated)."""
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("DISTIL_MAX_TOKENS_NOTE", "9999")
    assert resolve_stage_max_tokens("note") == 9999
    assert resolve_stage_max_tokens("extract") == 128_000
    assert resolve_stage_max_tokens("link") == 4096


@pytest.mark.unit
def test_max_tokens_env_override_can_lower_extracts_ceiling_too(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("DISTIL_MAX_TOKENS_EXTRACT", "2048")
    assert resolve_stage_max_tokens("extract") == 2048


@pytest.mark.unit
def test_make_stage_client_extract_gets_the_large_ceiling(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    client = make_stage_client("extract")
    assert client.max_tokens == 128_000


@pytest.mark.unit
def test_make_stage_client_non_extract_stage_gets_the_default_ceiling(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    client = make_stage_client("note")
    assert client.max_tokens == 4096


# ---- graph moved to the cheap tier (owner decision, Phase F) ---------------------------


@pytest.mark.unit
def test_graph_is_now_a_cheap_tier_stage():
    assert "graph" in CHEAP_TIER_STAGES
    assert "graph" not in STRONG_TIER_STAGES


@pytest.mark.unit
def test_graph_defaults_to_the_cheap_tier_without_any_env_var():
    assert resolve_stage_model("graph") == DEFAULT_SUMMARY_MODEL


@pytest.mark.unit
def test_graph_is_unaffected_by_distil_model(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL", "claude-opus-5")
    assert resolve_stage_model("graph") == DEFAULT_SUMMARY_MODEL


# ---- Settings-surface precedence: stored > DISTIL_MODEL_<STAGE> > tier default (T-MCFG6) --


@pytest.mark.unit
def test_stored_setting_overrides_the_stage_env_var_and_the_default(monkeypatch):
    from distil.model_settings import ModelSettingsStore

    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    ModelSettingsStore().set("extract", "claude-opus-5")
    assert resolve_stage_model("extract") == "claude-opus-5"


@pytest.mark.unit
def test_stored_setting_overrides_the_global_distil_model(monkeypatch):
    from distil.model_settings import ModelSettingsStore

    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    ModelSettingsStore().set("extract", "claude-opus-5")
    assert resolve_stage_model("extract") == "claude-opus-5"


@pytest.mark.unit
def test_stored_setting_overrides_the_cheap_tier_default(monkeypatch):
    from distil.model_settings import ModelSettingsStore

    ModelSettingsStore().set("triage", "claude-opus-5")
    assert resolve_stage_model("triage") == "claude-opus-5"


@pytest.mark.unit
def test_clearing_a_stored_setting_reverts_to_the_env_var(monkeypatch):
    from distil.model_settings import ModelSettingsStore

    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    settings = ModelSettingsStore()
    settings.set("extract", "claude-opus-5")
    settings.clear("extract")
    assert resolve_stage_model("extract") == "claude-sonnet-5"


@pytest.mark.unit
def test_clearing_a_stored_setting_reverts_to_the_tier_default(monkeypatch):
    from distil.model_settings import ModelSettingsStore

    settings = ModelSettingsStore()
    settings.set("triage", "claude-opus-5")
    settings.clear("triage")
    assert resolve_stage_model("triage") == DEFAULT_CHEAP_TIER_MODEL


@pytest.mark.unit
def test_a_stage_with_no_stored_setting_resolves_exactly_as_before(monkeypatch):
    """No ModelSettingsStore row anywhere for this stage — behavior must be identical to the
    pre-settings-surface resolution (env var / tier default), never a regression."""
    monkeypatch.setenv("DISTIL_MODEL", "claude-opus-5")
    assert resolve_stage_model("canonicalize") == "claude-opus-5"
    assert resolve_stage_model("triage") == DEFAULT_CHEAP_TIER_MODEL


# ---- resolve_stage_model_info: source reporting for the settings page ------------------


@pytest.mark.unit
def test_stage_info_reports_stored_source(monkeypatch):
    from distil.model_config import SOURCE_STORED, resolve_stage_model_info
    from distil.model_settings import ModelSettingsStore

    ModelSettingsStore().set("extract", "claude-opus-5")
    info = resolve_stage_model_info("extract")
    assert info.model == "claude-opus-5"
    assert info.source == SOURCE_STORED


@pytest.mark.unit
def test_stage_info_reports_env_source_for_the_per_stage_override(monkeypatch):
    from distil.model_config import SOURCE_ENV, resolve_stage_model_info

    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    info = resolve_stage_model_info("extract")
    assert info.model == "claude-sonnet-5"
    assert info.source == SOURCE_ENV
    assert info.active_env_var == "DISTIL_MODEL_EXTRACT"


@pytest.mark.unit
def test_stage_info_reports_env_source_for_the_global_distil_model_fallback(monkeypatch):
    from distil.model_config import SOURCE_ENV, resolve_stage_model_info

    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    info = resolve_stage_model_info("canonicalize")
    assert info.model == "claude-sonnet-5"
    assert info.source == SOURCE_ENV
    assert info.active_env_var == "DISTIL_MODEL"


@pytest.mark.unit
def test_stage_info_reports_default_source_for_a_cheap_tier_stage(monkeypatch):
    from distil.model_config import SOURCE_DEFAULT, resolve_stage_model_info

    info = resolve_stage_model_info("triage")
    assert info.model == DEFAULT_CHEAP_TIER_MODEL
    assert info.source == SOURCE_DEFAULT
    assert info.active_env_var is None


@pytest.mark.unit
def test_stage_info_reports_unconfigured_for_a_strong_stage_with_nothing_set():
    from distil.model_config import SOURCE_UNCONFIGURED, resolve_stage_model_info

    info = resolve_stage_model_info("extract")
    assert info.model is None
    assert info.source == SOURCE_UNCONFIGURED


@pytest.mark.unit
def test_stage_info_env_var_is_always_the_per_stage_name_regardless_of_source(monkeypatch):
    from distil.model_config import resolve_stage_model_info

    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    assert resolve_stage_model_info("canonicalize").env_var == "DISTIL_MODEL_CANONICALIZE"


@pytest.mark.unit
def test_stage_info_flags_the_faithfulness_warning_for_extract_on_haiku(monkeypatch):
    from distil.model_config import resolve_stage_model_info
    from distil.model_settings import ModelSettingsStore

    ModelSettingsStore().set("extract", "claude-haiku-4-5")
    info = resolve_stage_model_info("extract")
    assert info.faithfulness_warning is not None
    assert "8 of 16" in info.faithfulness_warning


@pytest.mark.unit
def test_stage_info_has_no_warning_for_extract_on_a_strong_model(monkeypatch):
    from distil.model_config import resolve_stage_model_info

    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    assert resolve_stage_model_info("extract").faithfulness_warning is None


@pytest.mark.unit
def test_stage_info_has_no_warning_for_a_non_extract_stage_on_haiku():
    from distil.model_config import resolve_stage_model_info

    info = resolve_stage_model_info("triage")  # cheap tier default is already Haiku
    assert info.faithfulness_warning is None


@pytest.mark.unit
def test_all_stages_covers_every_strong_and_cheap_tier_stage():
    from distil.model_config import ALL_STAGES

    assert set(ALL_STAGES) == {*STRONG_TIER_STAGES, *CHEAP_TIER_STAGES}
