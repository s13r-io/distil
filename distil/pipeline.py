"""Pipeline orchestration — wires stages 0→6. ARCHITECTURE.md §2; TESTING T-PL1, T-PL2.

One call turns a normalized transcript + profile into a filed, schema-valid :class:`KBEntry`:

    ingest (done by caller) → triage → [short-circuit] → extract → normalize → link
    → note synthesis → graph → file

The ``little_to_extract`` verdict short-circuits: a minimal entry is returned but **not filed**,
and no extract/link/graph LLM calls are made (T-PL2). The LLM-call budget is kept bounded
(triage + extract + link, plus capped graph calls).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from .embed import Embedder
from .extract import run_extraction
from .graph import link_graph
from .ingest import Transcript
from .link import generate_links
from .llm import LLMClient
from .models import EntryMeta, KBEntry, Profile, Source, Tags
from .normalize import normalize_items
from .note import synthesize_note
from .store import Store
from .triage import is_low_value, run_triage


@dataclass
class PipelineConfig:
    novelty_ratio: float = 0.2
    enable_graph: bool = True
    model_version: str = ""
    timing_callback: Callable[[str, float], None] | None = None


def run_pipeline(
    transcript: Transcript,
    profile: Profile,
    store: Store,
    client: LLMClient,
    *,
    source_title: str = "Untitled",
    source_url: str | None = None,
    source_channel: str | None = None,
    source_channel_url: str | None = None,
    source_thumbnail_url: str | None = None,
    source_metadata_provider: str | None = None,
    source_metadata_fetched_at: str | None = None,
    config: PipelineConfig | None = None,
    embedder: Embedder | None = None,
) -> KBEntry:
    config = config or PipelineConfig()
    entry_id = f"e_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    source = Source(
        url=source_url,
        title=source_title,
        channel=source_channel,
        channel_url=source_channel_url,
        thumbnail_url=source_thumbnail_url,
        metadata_provider=source_metadata_provider,
        metadata_fetched_at=source_metadata_fetched_at,
        captured_at=now,
    )
    meta = EntryMeta(created_at=now, model_version=config.model_version)

    # Stage 1 — triage (always one LLM call).
    triage_result = _timed("triage", config, lambda: run_triage(transcript, client))
    triage = triage_result.triage

    # Honesty short-circuit: return a minimal entry, no filing or further LLM calls (T-PL2).
    if is_low_value(triage_result):
        return KBEntry(entry_id=entry_id, source=source, triage=triage, meta=meta)

    # Stage 2 — extract; Stage 3 — normalize (pure faithfulness gate).
    raw_items = _timed("extract", config, lambda: run_extraction(transcript, triage, client))
    items = _timed("normalize", config, lambda: normalize_items(raw_items, transcript))

    # Stage 4 — link to profile.
    links = _timed(
        "link",
        config,
        lambda: generate_links(items, profile, client, novelty_ratio=config.novelty_ratio),
    )

    # Stage 5 — turn verified evidence into a reader-facing teaching note.
    distilled_note = _timed(
        "note",
        config,
        lambda: synthesize_note(source_title, triage, items, links, client),
    )

    entry = KBEntry(
        entry_id=entry_id,
        source=source,
        triage=triage,
        knowledge_items=items,
        application_links=links,
        distilled_note=distilled_note,
        tags=_derive_tags(items, links, distilled_note),
        meta=meta,
    )

    # Stage 6 — graph link against existing KB (capped; deterministic candidate lookup first).
    if config.enable_graph:
        entry.related_entries = _timed("graph", config, lambda: link_graph(entry, store, client))

    # Stage 7 — file (and embed items into the vector store for the read layer).
    _timed("file", config, lambda: store.file_entry(entry, embedder=embedder))
    return entry


def _derive_tags(items, links, note) -> Tags:
    types = sorted({it.type for it in items})
    forms = sorted({link.application_form for link in links})
    topics = note.topics if note is not None else []
    return Tags(topics=list(topics), knowledge_types=list(types), application_forms=list(forms))


def _timed(stage: str, config: PipelineConfig, fn):
    start = perf_counter()
    try:
        return fn()
    finally:
        if config.timing_callback is not None:
            config.timing_callback(stage, perf_counter() - start)
