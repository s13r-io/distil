"""Typer CLI — the full loop from the terminal. ARCHITECTURE.md §3; TESTING T-C1..C5.

Commands: ``run`` (ingest + pipeline), ``score`` (feedback → profile), ``list``/``show``
(browse), plus ``ask``/``reindex`` (added in Phase 10). Configuration comes from env; the LLM
client is built via :func:`_make_client`, a seam tests replace with a FakeClient.

Errors are surfaced as friendly messages (no stack traces) — a missing API key, a bad file, or
an unknown entry should read as guidance, not a crash.
"""

from __future__ import annotations

import os
import sys

import typer

from .ingest import IngestError, Transcript, ingest_file, ingest_text
from .llm import AnthropicClient, LLMClient
from .models import Profile
from .pipeline import PipelineConfig, run_pipeline
from .profile_update import apply_feedback
from .store import Store

app = typer.Typer(add_completion=False, help="Distil — personal knowledge distiller.")

_USER_ID = "owner"


# ---- dependency seams (tests monkeypatch these) ----------------------------------------


def _make_store() -> Store:
    db = os.environ.get("DISTIL_DB_PATH", "./data/distil.db")
    kb = os.environ.get("DISTIL_KB_DIR", "./kb")
    return Store(db_path=db, kb_dir=kb)


def _make_client() -> LLMClient:
    return AnthropicClient()


def _load_or_default_profile(store: Store) -> Profile:
    return store.load_profile(_USER_ID) or Profile(user_id=_USER_ID)


def _novelty_ratio() -> float:
    try:
        return float(os.environ.get("DISTIL_NOVELTY_RATIO", "0.2"))
    except ValueError:
        return 0.2


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=False)
    raise typer.Exit(code)


# ---- commands ---------------------------------------------------------------------------


@app.command()
def run(
    source: str = typer.Argument(None, help="Path to a .srt/.txt/.md transcript."),
    paste: str = typer.Option(None, "--paste", help="Pasted transcript text (instead of a file)."),
    no_graph: bool = typer.Option(False, "--no-graph", help="Skip cross-linking to existing KB."),
):
    """Ingest a transcript and run the distillation pipeline, filing a KB entry."""
    try:
        transcript: Transcript
        title = "Pasted transcript"
        if paste is not None:
            transcript = ingest_text(paste)
        elif source is not None:
            transcript = ingest_file(source)
            title = os.path.basename(source)
        elif not sys.stdin.isatty():
            transcript = ingest_text(sys.stdin.read())
        else:
            _fail("Provide a transcript: `distil run <file>` or `distil run --paste \"...\"`.")
            return
    except IngestError as exc:
        _fail(f"Could not read the transcript: {exc}")
        return

    store = _make_store()
    profile = _load_or_default_profile(store)
    try:
        client = _make_client()
        entry = run_pipeline(
            transcript, profile, store, client,
            source_title=title,
            config=PipelineConfig(novelty_ratio=_novelty_ratio(), enable_graph=not no_graph),
        )
    except RuntimeError as exc:
        # e.g. missing ANTHROPIC_API_KEY / DISTIL_MODEL
        _fail(f"{exc}")
        return

    typer.echo(str(store.entry_path(entry.entry_id)))


@app.command()
def score(
    entry_id: str = typer.Argument(..., help="The entry id to score."),
    score: int = typer.Option(..., "--score", min=1, max=5, help="1 (not useful) … 5 (very useful)."),
    reason: str = typer.Option(
        ..., "--reason",
        help="relevant | already_knew | bad_source | wrong_for_me | irrelevant_now",
    ),
):
    """Record feedback for an entry and update the profile."""
    store = _make_store()
    try:
        entry = store.load_entry(entry_id)
    except (FileNotFoundError, ValueError):
        _fail(f"Entry '{entry_id}' not found.")
        return

    entry.feedback.score = score
    try:
        entry.feedback.reason = reason  # validated on assignment via model
        entry = entry.model_validate(entry.model_dump())
    except Exception:
        _fail(
            "Invalid reason. Use one of: relevant, already_knew, bad_source, "
            "wrong_for_me, irrelevant_now."
        )
        return

    store.file_entry(entry)  # persist the feedback onto the entry
    profile = _load_or_default_profile(store)
    updated = apply_feedback(profile, entry, alpha=_alpha())
    store.save_profile(updated)
    typer.echo(f"Scored {entry_id}: {score} ({reason}). Profile updated.")


@app.command(name="list")
def list_entries():
    """List filed KB entries."""
    store = _make_store()
    rows = store.list_entries()
    if not rows:
        typer.echo("No entries yet. Run `distil run <transcript>` to add one.")
        return
    for r in rows:
        score = r.score if r.score is not None else "-"
        typer.echo(f"{r.entry_id}  [{score}]  {r.title}")


@app.command()
def show(entry_id: str = typer.Argument(..., help="The entry id to display.")):
    """Print a filed KB entry (markdown)."""
    store = _make_store()
    path = store.entry_path(entry_id)
    if not path.exists():
        _fail(f"Entry '{entry_id}' not found.")
        return
    typer.echo(path.read_text(encoding="utf-8"))


def _alpha() -> float:
    try:
        return float(os.environ.get("DISTIL_PROFILE_ALPHA", "0.3"))
    except ValueError:
        return 0.3


if __name__ == "__main__":  # pragma: no cover
    app()
