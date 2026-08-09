"""Durable per-stage model overrides (distil/model_settings.py) — T-MSET1..3 in docs/TESTING.md.
No network, no real database beyond a tmp_path sqlite file."""

import pytest

from distil.model_settings import ModelSettingsStore, UnknownModelError, is_known_model


@pytest.mark.unit
def test_get_on_a_missing_database_returns_none_without_creating_it(tmp_path):
    db_path = tmp_path / "distil.db"
    store = ModelSettingsStore(db_path)
    assert store.get("extract") is None
    assert not db_path.exists()


@pytest.mark.unit
def test_all_on_a_missing_database_returns_empty_without_creating_it(tmp_path):
    db_path = tmp_path / "distil.db"
    store = ModelSettingsStore(db_path)
    assert store.all() == {}
    assert not db_path.exists()


@pytest.mark.unit
def test_set_then_get_round_trips(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "claude-opus-5")
    assert store.get("extract") == "claude-opus-5"


@pytest.mark.unit
def test_set_overwrites_a_previous_value_for_the_same_stage(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "claude-opus-5")
    store.set("extract", "claude-sonnet-5")
    assert store.get("extract") == "claude-sonnet-5"


@pytest.mark.unit
def test_set_strips_surrounding_whitespace(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "  claude-opus-5  ")
    assert store.get("extract") == "claude-opus-5"


@pytest.mark.unit
def test_clear_removes_a_stored_setting(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "claude-opus-5")
    store.clear("extract")
    assert store.get("extract") is None


@pytest.mark.unit
def test_clear_on_a_stage_with_nothing_stored_is_a_no_op(tmp_path):
    """One obvious revert action, safe to click twice — never errors."""
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.clear("extract")  # no database yet
    assert store.get("extract") is None
    store.set("triage", "claude-haiku-4-5")
    store.clear("extract")  # database exists now, but no row for "extract"
    assert store.get("triage") == "claude-haiku-4-5"


@pytest.mark.unit
def test_clear_does_not_affect_other_stages(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "claude-opus-5")
    store.set("triage", "claude-haiku-4-5")
    store.clear("extract")
    assert store.get("extract") is None
    assert store.get("triage") == "claude-haiku-4-5"


@pytest.mark.unit
def test_set_an_unknown_model_raises_and_writes_nothing(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    with pytest.raises(UnknownModelError):
        store.set("extract", "gpt-5")
    assert store.get("extract") is None


@pytest.mark.unit
def test_unknown_model_error_names_the_bad_string(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    with pytest.raises(UnknownModelError, match="not-a-real-model"):
        store.set("extract", "not-a-real-model")


@pytest.mark.unit
def test_is_known_model():
    assert is_known_model("claude-sonnet-5")
    assert not is_known_model("not-a-real-model")


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5", "claude-fable-5", "claude-opus-4-8"],
)
def test_known_models_are_accepted(tmp_path, model):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", model)  # must not raise
    assert store.get("extract") == model


@pytest.mark.unit
def test_setting_survives_a_fresh_store_instance_pointed_at_the_same_path(tmp_path):
    """Durability isn't an artifact of one Python object staying alive — a fresh instance
    (e.g. a new request, or a restarted process) must see it too."""
    db_path = tmp_path / "distil.db"
    ModelSettingsStore(db_path).set("extract", "claude-opus-5")
    assert ModelSettingsStore(db_path).get("extract") == "claude-opus-5"


@pytest.mark.unit
def test_all_reports_every_stored_stage(tmp_path):
    store = ModelSettingsStore(tmp_path / "distil.db")
    store.set("extract", "claude-opus-5")
    store.set("triage", "claude-haiku-4-5")
    rows = store.all()
    assert set(rows) == {"extract", "triage"}
    assert rows["extract"].model == "claude-opus-5"
    assert rows["extract"].updated_at  # non-empty ISO timestamp


@pytest.mark.unit
def test_default_db_path_reads_distil_db_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "elsewhere.db"))
    store = ModelSettingsStore()
    assert store.db_path == tmp_path / "elsewhere.db"
