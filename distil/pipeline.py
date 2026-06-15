"""Pipeline orchestration — wires stages 0→6. ARCHITECTURE.md §2; TESTING T-PL1, T-PL2.

One call turns a normalized transcript + profile into a filed, schema-valid :class:`KBEntry`:

    ingest (done by caller) → triage → [short-circuit] → extract → normalize → link → graph → file

The ``little_to_extract`` verdict short-circuits: a minimal entry is filed and **no**
extract/link/graph LLM calls are made (T-PL2). The LLM-call budget is kept bounded
(triage + extract + link, plus capped graph calls).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .embed import Embedder
from .extract import run_extraction
from .graph import link_graph
from .ingest import Transcript
from .link import generate_links
from .llm import LLMClient
from .models import EntryMeta, KBEntry, Profile, Source, Tags
from .normalize import normalize_items
from .store import Store
from .triage import is_low_value, run_triage


@dataclass
class PipelineConfig:
    novelty_ratio: float = 0.2
    enable_graph: bool = True
    model_version: str = ""


def run_pipeline(
    transcript: Transcript,
    profile: Profile,
    store: Store,
    client: LLMClient,
    *,
    source_title: str = "Untitled",
    source_url: str | None = None,
    config: PipelineConfig | None = None,
    embedder: Embedder | None = None,
) -> KBEntry:
    config = config or PipelineConfig()
    entry_id = f"e_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    source = Source(url=source_url, title=source_title, captured_at=now)
    meta = EntryMeta(created_at=now, model_version=config.model_version)

    # Stage 1 — triage (always one LLM call).
    triage_result = run_triage(transcript, client)
    triage = triage_result.triage

    # Honesty short-circuit: file a minimal entry, no further LLM calls (T-PL2).
    if is_low_value(triage_result):
        entry = KBEntry(entry_id=entry_id, source=source, triage=triage, meta=meta)
        store.file_entry(entry, embedder=embedder)  # no items → no vectors
        return entry

    # Stage 2 — extract; Stage 3 — normalize (pure faithfulness gate).
    raw_items = run_extraction(transcript, triage, client)
    items = normalize_items(raw_items, transcript)

    # Stage 4 — link to profile.
    links = generate_links(items, profile, client, novelty_ratio=config.novelty_ratio)

    entry = KBEntry(
        entry_id=entry_id,
        source=source,
        triage=triage,
        knowledge_items=items,
        application_links=links,
        tags=_derive_tags(items, links),
        meta=meta,
    )

    # Stage 5 — graph link against existing KB (capped; deterministic candidate lookup first).
    if config.enable_graph:
        entry.related_entries = link_graph(entry, store, client)

    # Stage 6 — file (and embed items into the vector store for the read layer).
    store.file_entry(entry, embedder=embedder)
    return entry


def _derive_tags(items, links) -> Tags:
    types = sorted({it.type for it in items})
    forms = sorted({link.application_form for link in links})
    # Topics aren't a first-class field on items; derive a light tag set from item types for
    # now (richer topic tagging can be added without schema change).
    return Tags(topics=[], knowledge_types=list(types), application_forms=list(forms))
