"""Model settings surface (/settings, /settings/models/*) — the web UI half of the durable
per-stage model override described in distil/model_config.py and distil/model_settings.py.

Behavioral tests against the real routes via TestClient, not template-source assertions — this
project has already rejected source-grep tests once (see AGENTS.md) and forced the resolution
logic server-side, in distil/model_config.py, which is what these tests actually exercise.
"""

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from distil.models import Profile
    from distil.store import Store

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    monkeypatch.delenv("DISTIL_MODEL", raising=False)
    monkeypatch.delenv("DISTIL_MODEL_EXTRACT", raising=False)
    monkeypatch.delenv("DISTIL_MODEL_TRIAGE", raising=False)
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    return TestClient(create_app())


@pytest.mark.unit
def test_settings_page_loads(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "Model settings" in r.text


@pytest.mark.unit
def test_settings_page_requires_auth_when_public(tmp_path, monkeypatch):
    from distil.models import Profile
    from distil.store import Store

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cr3t")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    test_client = TestClient(create_app())
    r = test_client.get("/settings/models", headers={"accept": "application/json"})
    assert r.status_code == 401


@pytest.mark.unit
def test_settings_models_lists_every_stage(client):
    from distil.model_config import ALL_STAGES

    body = client.get("/settings/models").json()
    stages = {row["stage"] for row in body["stages"]}
    assert stages == set(ALL_STAGES)


@pytest.mark.unit
def test_settings_models_reports_default_source_for_a_cheap_tier_stage(client):
    from distil.model_config import DEFAULT_CHEAP_TIER_MODEL

    body = client.get("/settings/models").json()
    row = next(r for r in body["stages"] if r["stage"] == "triage")
    assert row["source"] == "default"
    assert row["model"] == DEFAULT_CHEAP_TIER_MODEL
    assert row["has_stored_override"] is False


@pytest.mark.unit
def test_settings_models_reports_env_source_for_the_global_distil_model(client, monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL", "claude-sonnet-5")
    body = client.get("/settings/models").json()
    row = next(r for r in body["stages"] if r["stage"] == "canonicalize")
    assert row["source"] == "env"
    assert row["model"] == "claude-sonnet-5"
    assert row["active_env_var"] == "DISTIL_MODEL"


@pytest.mark.unit
def test_settings_models_reports_unconfigured_for_a_strong_stage_with_nothing_set(client):
    body = client.get("/settings/models").json()
    row = next(r for r in body["stages"] if r["stage"] == "extract")
    assert row["source"] == "unconfigured"
    assert row["model"] is None


@pytest.mark.unit
def test_post_valid_model_stores_it_and_survives_a_fresh_lookup(client):
    r = client.post("/settings/models/extract", data={"model": "claude-opus-5"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "claude-opus-5"
    assert body["source"] == "stored"

    # Fresh read, not just the echoed response — proves it's actually durable.
    row = next(
        r for r in client.get("/settings/models").json()["stages"] if r["stage"] == "extract"
    )
    assert row["model"] == "claude-opus-5"
    assert row["source"] == "stored"
    assert row["has_stored_override"] is True


@pytest.mark.unit
def test_stored_setting_overrides_the_env_var(client, monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    client.post("/settings/models/extract", data={"model": "claude-opus-5"})
    row = next(
        r for r in client.get("/settings/models").json()["stages"] if r["stage"] == "extract"
    )
    assert row["model"] == "claude-opus-5"
    assert row["source"] == "stored"


@pytest.mark.unit
def test_post_unknown_model_is_refused_with_a_clear_reason_and_nothing_is_stored(client):
    r = client.post("/settings/models/extract", data={"model": "gpt-5-turbo"})
    assert r.status_code == 400
    assert "gpt-5-turbo" in r.json()["detail"]

    row = next(
        r for r in client.get("/settings/models").json()["stages"] if r["stage"] == "extract"
    )
    assert row["source"] == "unconfigured"  # nothing was written by the refused request


@pytest.mark.unit
def test_post_unknown_stage_name_is_rejected(client):
    r = client.post("/settings/models/not-a-real-stage", data={"model": "claude-opus-5"})
    assert r.status_code == 404


@pytest.mark.unit
def test_clear_reverts_to_the_env_var(client, monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL_EXTRACT", "claude-sonnet-5")
    client.post("/settings/models/extract", data={"model": "claude-opus-5"})
    r = client.post("/settings/models/extract/clear")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "claude-sonnet-5"
    assert body["source"] == "env"


@pytest.mark.unit
def test_clear_reverts_to_the_cheap_tier_default(client):
    from distil.model_config import DEFAULT_CHEAP_TIER_MODEL

    client.post("/settings/models/triage", data={"model": "claude-opus-5"})
    r = client.post("/settings/models/triage/clear")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == DEFAULT_CHEAP_TIER_MODEL
    assert body["source"] == "default"


@pytest.mark.unit
def test_clear_on_a_stage_with_nothing_stored_is_a_harmless_no_op(client):
    r = client.post("/settings/models/extract/clear")
    assert r.status_code == 200
    assert r.json()["source"] == "unconfigured"


@pytest.mark.unit
def test_setting_extract_to_a_cheap_model_warns_but_does_not_block(client):
    r = client.post("/settings/models/extract", data={"model": "claude-haiku-4-5"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "claude-haiku-4-5"  # not blocked
    assert body["faithfulness_warning"] is not None
    assert "8 of 16" in body["faithfulness_warning"]


@pytest.mark.unit
def test_setting_extract_to_a_strong_model_has_no_warning(client):
    r = client.post("/settings/models/extract", data={"model": "claude-opus-5"})
    assert r.json()["faithfulness_warning"] is None


@pytest.mark.unit
def test_setting_a_non_extract_stage_to_haiku_has_no_faithfulness_warning(client):
    r = client.post("/settings/models/triage", data={"model": "claude-haiku-4-5"})
    assert r.json()["faithfulness_warning"] is None


@pytest.mark.unit
def test_no_route_ever_mentions_an_api_key(client, monkeypatch):
    """This surface selects models only — never touches ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    body = client.get("/settings/models").json()
    assert "sk-ant-super-secret-value" not in str(body)
    page = client.get("/settings").text
    assert "sk-ant-super-secret-value" not in page
    assert "api_key" not in page.lower().replace("apikey", "")


@pytest.mark.unit
def test_setting_stored_at_a_path_is_read_back_by_a_fresh_app_instance(client, tmp_path):
    """Survives 'restart' — a brand-new create_app() (fresh Worker/Fetcher, fresh everything)
    reading the same DISTIL_DB_PATH must see the stored setting."""
    client.post("/settings/models/extract", data={"model": "claude-opus-5"})
    fresh_client = TestClient(create_app())
    row = next(
        r for r in fresh_client.get("/settings/models").json()["stages"]
        if r["stage"] == "extract"
    )
    assert row["model"] == "claude-opus-5"
    assert row["source"] == "stored"
