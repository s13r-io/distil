"""Phase 23 — /diagnostics/youtube-pot: the permanent PO-token diagnostic, usable over HTTP
with no shell/filesystem access to the running service.

`distil.youtube.diagnose_pot` itself is unit-tested against an injected `run` callable in
tests/unit/test_youtube.py; here we only cover route wiring (auth, URL validation, JSON shape),
so `web.app.youtube.diagnose_pot` is monkeypatched rather than touching the real subprocess
boundary.
"""

import pytest
from fastapi.testclient import TestClient

from distil.youtube import PotDiagnostic
from web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from distil.models import Profile
    from distil.store import Store
    from web import app as webapp

    monkeypatch.setenv("DISTIL_DB_PATH", str(tmp_path / "distil.db"))
    monkeypatch.setenv("DISTIL_KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    Store(db_path=tmp_path / "distil.db", kb_dir=tmp_path / "kb").save_profile(
        Profile(user_id="owner")
    )
    return TestClient(create_app()), webapp


@pytest.mark.unit
def test_diagnose_youtube_pot_rejects_non_youtube_url(client):
    test_client, _ = client
    r = test_client.get("/diagnostics/youtube-pot", params={"url": "https://example.com/x"})
    assert r.status_code == 400


@pytest.mark.unit
def test_diagnose_youtube_pot_reports_no_attempts(client):
    test_client, webapp = client
    webapp.youtube.diagnose_pot = lambda url, **kwargs: PotDiagnostic(
        returncode=1, provider_discovery=None, context_attempts=[], raw_output="ERROR: bot check"
    )
    r = test_client.get(
        "/diagnostics/youtube-pot", params={"url": "https://www.youtube.com/watch?v=abc"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provider_discovery"] is None
    assert body["context_attempts"] == []
    assert "ERROR: bot check" in body["raw_output"]


@pytest.mark.unit
def test_diagnose_youtube_pot_reports_context_attempts(client):
    test_client, webapp = client
    webapp.youtube.diagnose_pot = lambda url, **kwargs: PotDiagnostic(
        returncode=0,
        provider_discovery="[youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external)",
        context_attempts=[("player", "mweb")],
        raw_output="...",
    )
    r = test_client.get(
        "/diagnostics/youtube-pot", params={"url": "https://www.youtube.com/watch?v=abc"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provider_discovery"] == "[youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external)"
    assert body["context_attempts"] == [{"context": "player", "client": "mweb"}]


@pytest.mark.unit
def test_diagnose_youtube_pot_requires_auth_when_public(tmp_path, monkeypatch):
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
    r = test_client.get(
        "/diagnostics/youtube-pot",
        params={"url": "https://www.youtube.com/watch?v=abc"},
        headers={"accept": "application/json"},
    )
    assert r.status_code == 401
