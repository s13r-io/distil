"""Phase 11.2 / 12 — auth gate + web app. Tests T-A1..A4 + basic UI routes.

The auth gate is security-critical: the app must FAIL CLOSED when exposed publicly without a
secret, and never serve data to an unauthenticated request when public.
"""

import pytest
from fastapi.testclient import TestClient

from distil.models import KBEntry, Profile
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
def test_a2_public_data_route_redirects_browser_to_login(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app(), follow_redirects=False)
    r = client.get("/entries", headers={"accept": "text/html"})
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@pytest.mark.unit
def test_a2_public_data_route_returns_401_for_api(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    r = client.get("/entries", headers={"accept": "application/json"})
    assert r.status_code == 401


@pytest.mark.unit
def test_a2_public_data_route_with_bearer_ok(seeded, monkeypatch):
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
    assert "Ask your notes" in r.text
    assert "Add knowledge" in r.text
    assert 'href="/library"' in r.text


@pytest.mark.unit
def test_library_page_renders(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    client = TestClient(create_app())
    r = client.get("/library")
    assert r.status_code == 200
    assert "Saved notes" in r.text
    assert "Search titles or tags" in r.text
    assert "1 star" in r.text
    assert "2 star" in r.text


@pytest.mark.unit
def test_library_page_renders_collapsed_humanized_tag_cloud(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    entry = KBEntry.model_validate({
        "entry_id": "e_tags",
        "source": {"title": "Tagged", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [],
        "tags": {"topics": ["ai_agent_memory"], "knowledge_types": ["heuristic"]},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    })
    store.file_entry(entry)
    client = TestClient(create_app())
    r = client.get("/library")
    assert r.status_code == 200
    assert "<summary>Content tags</summary>" in r.text
    assert ">AI Agent Memory</button>" in r.text
    assert ">ai_agent_memory</button>" not in r.text


@pytest.mark.unit
def test_entry_page_renders_distilled_note(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.file_entry(KBEntry.model_validate({
        "entry_id": "e_note",
        "source": {
            "title": "A talk",
            "url": "https://youtu.be/abc123",
            "channel": "Talk Channel",
            "channel_url": "https://www.youtube.com/@talk",
            "thumbnail_url": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
            "captured_at": "2026-06-15T00:00:00",
        },
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01",
            "type": "heuristic",
            "statement": "Keep functions small.",
            "stance": "opinion",
            "provenance": {"quote": "keep functions small"},
        }],
        "distilled_note": {
            "title": "Small functions",
            "core_takeaway": {
                "text": "Small functions are easier to understand.",
                "item_ids": ["k_01"],
            },
            "key_points": [{"text": "Keep one behavior per function.", "item_ids": ["k_01"]}],
            "topics": ["function_design"],
        },
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }))
    client = TestClient(create_app())
    r = client.get("/entries/e_note")
    assert r.status_code == 200
    assert "Teaching note" in r.text
    assert "Core takeaway" in r.text
    assert "Source evidence" in r.text
    assert 'aria-label="Copy markdown"' in r.text
    assert 'aria-label="Download markdown"' in r.text
    assert 'href="/entries/e_note/teaching-note.md?download=true"' in r.text
    assert "Watch on YouTube" in r.text
    assert "Talk Channel" in r.text
    assert "hqdefault.jpg" in r.text
    assert ">Tags</h2>" in r.text
    assert ">Topics</h2>" not in r.text
    assert "Function Design" in r.text
    assert "function_design" not in r.text

    md = client.get("/entries/e_note/teaching-note.md")
    assert md.status_code == 200
    assert "# Small functions" in md.text
    assert "## Metadata" in md.text
    assert "- Source URL: https://youtu.be/abc123" in md.text
    assert "## Core takeaway" in md.text
    assert "## Tags" in md.text
    assert "Function Design" in md.text
    assert "k_01" not in md.text

    download = client.get("/entries/e_note/teaching-note.md?download=true")
    assert download.status_code == 200
    assert download.headers["content-disposition"] == 'attachment; filename="Small-functions.md"'


@pytest.mark.unit
def test_legacy_entry_page_renders_markdown_icons_in_knowledge_header(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.file_entry(KBEntry.model_validate({
        "entry_id": "e_legacy",
        "source": {"title": "Legacy note", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01",
            "type": "heuristic",
            "statement": "Keep functions small.",
            "stance": "opinion",
            "provenance": {"quote": "keep functions small"},
        }],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }))
    client = TestClient(create_app())
    r = client.get("/entries/e_legacy")
    assert r.status_code == 200
    assert "Knowledge / 1 item" in r.text
    assert 'aria-label="Copy markdown"' in r.text
    assert 'aria-label="Download markdown"' in r.text
    assert 'href="/entries/e_legacy/teaching-note.md?download=true"' in r.text
    assert "<p class=\"small-title\">Markdown</p>" not in r.text


@pytest.mark.unit
def test_entry_delete_route_removes_entry(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "false")
    store = Store(db_path=seeded / "distil.db", kb_dir=seeded / "kb")
    store.file_entry(KBEntry.model_validate({
        "entry_id": "e_delete",
        "source": {"title": "Delete me", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [],
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }))
    client = TestClient(create_app(), follow_redirects=False)
    r = client.post("/entries/e_delete/delete")
    assert r.status_code == 303
    assert r.headers["location"] == "/library"
    assert not store.entry_path("e_delete").exists()


# ---- login / logout flow ----


@pytest.mark.unit
def test_login_page_renders(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    r = client.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text


@pytest.mark.unit
def test_login_correct_secret_sets_cookie_and_redirects(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app(), follow_redirects=False)
    r = client.post("/login", data={"secret": "s3cret"})
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert "distil_session" in r.cookies


@pytest.mark.unit
def test_login_wrong_secret_shows_error(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    client = TestClient(create_app())
    r = client.post("/login", data={"secret": "wrongpassword"})
    assert r.status_code == 200
    assert "Incorrect secret" in r.text


@pytest.mark.unit
def test_session_cookie_grants_access(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    # Use https base URL so httpx honours the Secure cookie flag (mirrors Railway HTTPS)
    client = TestClient(create_app(), follow_redirects=False, base_url="https://testserver")
    # Log in to get the cookie
    login = client.post("/login", data={"secret": "s3cret"})
    assert "distil_session" in login.cookies
    # Cookie is automatically sent on subsequent requests by TestClient
    r = client.get("/entries")
    assert r.status_code == 200


@pytest.mark.unit
def test_logout_clears_cookie(seeded, monkeypatch):
    monkeypatch.setenv("DISTIL_PUBLIC", "true")
    monkeypatch.setenv("DISTIL_AUTH_SECRET", "s3cret")
    # Use https base URL so httpx honours the Secure cookie flag (mirrors Railway HTTPS)
    client = TestClient(create_app(), follow_redirects=False, base_url="https://testserver")
    client.post("/login", data={"secret": "s3cret"})
    r = client.get("/logout")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
