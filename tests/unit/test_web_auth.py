"""Phase 11.2 / 12 — auth gate + web app. Tests T-A1..A4 + basic UI routes.

The auth gate is security-critical: the app must FAIL CLOSED when exposed publicly without a
secret, and never serve data to an unauthenticated request when public.
"""

import pytest
from fastapi.testclient import TestClient

from distil.models import Profile
from distil.store import Store
from web import auth
from web.app import create_app


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_MODEL", "test")
    s = Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb")
    s.save_profile(Profile(user_id="owner"))
    return tmp_path


# ---- T-A1: public mode without a secret fails closed ----


@pytest.mark.unit
def test_a1_public_without_secret_refuses_to_start(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.delenv("DISTIL_AUTH_SECRET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        create_app()
    assert "DISTIL_AUTH_SECRET" in str(exc.value)


@pytest.mark.unit
def test_a1_public_with_secret_starts(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    app = create_app()  # must not raise
    assert app is not None


# ---- T-A2: unauthenticated request to a data route returns 401 in public mode ----


@pytest.mark.unit
def test_a2_public_data_route_requires_auth(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    r = client.get("/entries")
    assert r.status_code == 401
    # never leaks data
    assert "entry" not in r.text.lower() or "unauthor" in r.text.lower()


@pytest.mark.unit
def test_a2_public_data_route_with_secret_ok(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    r = client.get("/entries", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


@pytest.mark.unit
def test_a2_health_is_open_even_in_public_mode(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200


# ---- T-A3: localhost (not public) reachable without the secret ----


@pytest.mark.unit
def test_a3_localhost_no_secret_reachable(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    monkeypatch.delenv("DISTIL_AUTH_SECRET", raising=False)
    client = TestClient(create_app())
    assert client.get("/entries").status_code == 200


# ---- T-A4: binds 0.0.0.0:$PORT readiness ----


@pytest.mark.unit
def test_a4_bind_config_uses_injected_port(monkeypatch):
    monkeypatch.setenv("PORT", "8123")
    host, port = auth.server_bind()
    assert host == "0.0.0.0"
    assert port == 8123


@pytest.mark.unit
def test_a4_bind_defaults_when_no_port(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    host, port = auth.server_bind()
    assert host == "0.0.0.0"
    assert isinstance(port, int)


# ---- basic UI route ----


@pytest.mark.unit
def test_index_page_renders(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "Distil" in r.text
