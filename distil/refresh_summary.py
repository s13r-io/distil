"""Per-entry narrative-summary refresh — generate or regenerate *only* ``narrative_summary``
for one already-filed entry, from its stored raw transcript (``okf/raw/<slug>.md``).

This is the one orchestration entry point both the CLI (``distil refresh-summary``) and the web
route (``POST /entries/{id}/refresh-summary``) call, mirroring how ``canonicalize.py`` gives
``run_canonicalize_stage``/``run_delete_entry_stage`` as the single place that talks to both
``store.py`` and ``okf.py`` for their respective operations.

Constraints that matter here (see the task brief, not just the code): never re-fetch from
YouTube (only reads the already-stored ``okf/raw/`` page), never re-run extraction, and never
touch concepts/entities/items. ``store.file_entry(entry)`` called with no ``transcript=`` kwarg
is exactly the "feedback-only re-file" path ``store.py`` already documents — it rewrites
``kb/<id>.md`` and the sqlite index row only, leaving the OKF ``sources/``/``raw/``/``concepts/``/
``entities/`` pages and every vector/membership untouched. That existing behavior is what makes
this refresh safe by construction, not something this module has to re-implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from . import okf
from .llm import LLMClient
from .models import NarrativeSummary
from .store import Store
from .summary import NarrativeSummaryError, synthesize_narrative_summary


@dataclass
class RefreshResult:
    ok: bool
    message: str


def refresh_narrative_summary(entry_id: str, store: Store, client: LLMClient) -> RefreshResult:
    """Regenerate ``entry.narrative_summary`` from the entry's stored raw transcript.

    ``client`` should be the cheap-tier summary client (same one the pipeline stage uses) —
    this function makes no model-tier decision of its own.
    """
    if not store.entry_path(entry_id).exists():
        return RefreshResult(False, f"Entry '{entry_id}' not found.")

    entry = store.load_entry(entry_id)
    slug = okf.slug_for_entry(entry, store.okf_root)
    transcript_text = okf.load_raw_transcript_text(slug, store.okf_root)
    if transcript_text is None:
        return RefreshResult(
            False,
            "No stored transcript is available for this entry, so a narrative summary can't "
            "be generated. This can happen for an entry filed before transcript storage "
            "existed, or if its raw transcript page was later removed. Refreshing never "
            "re-fetches from YouTube, so this isn't something retrying will fix.",
        )

    try:
        result = synthesize_narrative_summary(transcript_text, client)
    except NarrativeSummaryError as exc:
        return RefreshResult(False, f"Could not generate a narrative summary: {exc}")
    except Exception as exc:  # an unexpected failure still reports plainly, never a stack trace
        return RefreshResult(False, f"Could not generate a narrative summary: {exc}")

    entry.narrative_summary = NarrativeSummary(
        text=result.text,
        chunk_count=result.chunk_count,
        model=result.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    store.file_entry(entry)  # no transcript= -> items/concepts/entities/OKF pages untouched
    return RefreshResult(True, "Narrative summary regenerated.")
