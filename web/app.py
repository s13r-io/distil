"""FastAPI app (v0.2): list/view/score entries + an ask box over the read layer.

Auth middleware sits in front of all data routes (PRD FR14, T-A2); ``/health`` is open. The
app fails closed at construction when public without a secret (T-A1).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from distil.cli import _make_client, _make_embedder  # reuse seams
from distil.profile_update import apply_feedback
from distil.query import ask as run_ask
from distil.store import Store

from . import auth

_USER_ID = "owner"


def _store() -> Store:
    db = os.environ.get("DISTIL_DB_PATH", "./data/distil.db")
    kb = os.environ.get("DISTIL_KB_DIR", "./kb")
    return Store(db_path=db, kb_dir=kb)


def create_app() -> FastAPI:
    auth.assert_startup_safe()  # fail closed before serving (T-A1)
    app = FastAPI(title="Distil", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        if not auth.path_is_open(request.url.path):
            if not auth.request_is_authorized(request.headers.get("Authorization")):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index():
        rows = _store().list_entries()
        items = "".join(
            f'<li><a href="/entries/{r.entry_id}">{r.title}</a> '
            f'[{r.score if r.score is not None else "-"}]</li>'
            for r in rows
        )
        return (
            "<!doctype html><html><head><title>Distil</title></head><body>"
            "<h1>Distil — your knowledge base</h1>"
            '<form action="/ask" method="get"><input name="q" placeholder="Ask your notes…">'
            '<button>Ask</button></form>'
            f"<ul>{items or '<li>No entries yet.</li>'}</ul>"
            "</body></html>"
        )

    @app.get("/entries")
    def entries():
        return [
            {"entry_id": r.entry_id, "title": r.title, "score": r.score}
            for r in _store().list_entries()
        ]

    @app.get("/entries/{entry_id}")
    def entry(entry_id: str):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return JSONResponse({"detail": "not found"}, status_code=404)
        return JSONResponse(store.load_entry(entry_id).model_dump())

    @app.post("/entries/{entry_id}/score")
    def score(entry_id: str, score: int, reason: str):
        store = _store()
        if not store.entry_path(entry_id).exists():
            return JSONResponse({"detail": "not found"}, status_code=404)
        e = store.load_entry(entry_id)
        e.feedback.score = score
        try:
            e.feedback.reason = reason
            e = e.model_validate(e.model_dump())
        except Exception:
            return JSONResponse({"detail": "invalid reason"}, status_code=400)
        store.file_entry(e)
        profile = store.load_profile(_USER_ID) or _default_profile()
        store.save_profile(apply_feedback(profile, e))
        return {"ok": True}

    @app.get("/ask")
    def ask(q: str):
        store = _store()
        result = run_ask(q, store, _make_embedder(), _make_client())
        return {
            "abstained": result.abstained,
            "message": result.message,
            "answer": result.answer,
            "conflict": result.conflict,
            "sources": [
                {"entry_id": s.entry_id, "item_id": s.item_id,
                 "quote": s.quote, "timestamp": s.timestamp}
                for s in result.sources
            ],
        }

    return app


def _default_profile():
    from distil.models import Profile

    return Profile(user_id=_USER_ID)


# Module-level app for `uvicorn web.app:app` (railway.toml startCommand).
# Constructed lazily so importing the module doesn't fail-closed during tests.
def __getattr__(name):  # pragma: no cover
    if name == "app":
        return create_app()
    raise AttributeError(name)
