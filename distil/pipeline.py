"""Pipeline orchestration — wires stages 0→9. ARCHITECTURE.md §2; TESTING T-PL1, T-PL2, T-PL5, T-PL6.

One call turns a normalized transcript + profile into a filed, schema-valid :class:`KBEntry`:

    ingest (done by caller) → triage → [short-circuit] → extract → normalize → link
    → note synthesis → narrative summary → graph → file → canonicalize → concept edges

The ``little_to_extract`` verdict short-circuits: a minimal entry is returned but **not filed**,
and no extract/link/graph/canonicalize/concept-edge LLM calls are made (T-PL2). The LLM-call
budget is kept bounded (triage + extract + link, plus capped graph calls and capped
canonicalize/synthesis/concept-edge calls — see ``canonicalize.py``'s and ``concept_graph.py``'s
module docstrings and the OKF Phase 3 design report §6, §9 item 4).

The narrative summary stage (``distil/summary.py``) is additive and optional: it only runs when
a caller passes ``summary_client`` (a *separate*, cheap-tier client — never ``client``, which
stays on the strong model for triage/extract/link/note/canonicalize/concept-edges unchanged) and
``config.enable_narrative_summary`` is true. Omitting ``summary_client`` — every caller that
existed before this stage was added — reproduces the exact prior behavior: ``entry.
narrative_summary`` stays ``None`` and no extra LLM call is made. A failure inside the stage
(thin output that exhausted its retries, a dropped connection) is caught and logged rather than
propagated, so this layer can never turn a would-have-succeeded filing into a failed one.

Per-stage model selection (``distil/model_config.py``) is a general mechanism, not something
built one-off for the narrative summary: ``triage_client``/``extract_client``/``link_client``/
``note_client``/``graph_client``/``canonicalize_client`` let a caller inject a distinct client
per stage, each defaulting to ``None`` — meaning "use ``client``", the exact single shared
object every stage used before these existed. No current caller (``cli.py``, ``web/app.py``)
passes them yet, so today's actual model selection for these six stages is unchanged; that
wiring, and any settings UI for it, are deliberate follow-ups. What already works today is that
setting ``DISTIL_MODEL_<STAGE>`` in the environment and passing the matching
``model_config.make_stage_client(stage)`` result through one of these parameters changes only
that stage — no further ``pipeline.py`` change required.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from .canonicalize import run_canonicalize_stage
from .concept_graph import run_concept_edges_stage
from .embed import Embedder
from .extract import run_extraction
from .graph import link_graph
from .ingest import Transcript
from .link import generate_links
from .llm import LLMClient
from .models import EntryMeta, KBEntry, NarrativeSummary, Profile, Source, Tags
from .normalize import normalize_items
from .note import synthesize_note
from .store import Store
from .summary import NarrativeSummaryError, synthesize_narrative_summary
from .triage import is_low_value, run_triage

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    novelty_ratio: float = 0.2
    enable_graph: bool = True
    enable_canonicalize: bool = True
    enable_concept_edges: bool = True
    enable_entities: bool = True
    enable_narrative_summary: bool = True
    model_version: str = ""
    timing_callback: Callable[[str, float], None] | None = None
    # Reports stage start/finish (and the little_to_extract short-circuit) for live progress
    # display, independent of timing_callback's after-the-fact duration reporting. Events:
    # ("<stage>", "start"), ("<stage>", "finish"), ("triage", "short_circuit").
    phase_callback: Callable[[str, str], None] | None = None


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
    summary_client: LLMClient | None = None,
    triage_client: LLMClient | None = None,
    extract_client: LLMClient | None = None,
    link_client: LLMClient | None = None,
    note_client: LLMClient | None = None,
    graph_client: LLMClient | None = None,
    canonicalize_client: LLMClient | None = None,
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
    triage_result = _timed(
        "triage", config, lambda: run_triage(transcript, triage_client or client)
    )
    triage = triage_result.triage

    # Honesty short-circuit: return a minimal entry, no filing or further LLM calls (T-PL2).
    # Tell the phase reporter the run is stopping now, so a declared total sized for the full
    # sequence never gets reported as "stuck" on a run that will never reach it.
    if is_low_value(triage_result):
        if config.phase_callback is not None:
            config.phase_callback("triage", "short_circuit")
        return KBEntry(entry_id=entry_id, source=source, triage=triage, meta=meta)

    # Stage 2 — extract; Stage 3 — normalize (pure faithfulness gate).
    raw_items = _timed(
        "extract", config, lambda: run_extraction(transcript, triage, extract_client or client)
    )
    items = _timed("normalize", config, lambda: normalize_items(raw_items, transcript))

    # Stage 4 — link to profile.
    links = _timed(
        "link",
        config,
        lambda: generate_links(
            items, profile, link_client or client, novelty_ratio=config.novelty_ratio
        ),
    )

    # Stage 5 — turn verified evidence into a reader-facing teaching note.
    distilled_note = _timed(
        "note",
        config,
        lambda: synthesize_note(source_title, triage, items, links, note_client or client),
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

    # Stage 5.5 — narrative summary: reads the transcript directly (additive, cheap-tier;
    # see module docstring). No-op unless the caller opted in with a summary_client.
    if config.enable_narrative_summary and summary_client is not None:
        entry.narrative_summary = _timed(
            "narrative_summary",
            config,
            lambda: _safe_narrative_summary(transcript, summary_client),
        )

    # Stage 6 — graph link against existing KB (capped; deterministic candidate lookup first).
    if config.enable_graph:
        entry.related_entries = _timed(
            "graph", config, lambda: link_graph(entry, store, graph_client or client)
        )

    # Stage 7 — file (and embed items into the vector store for the read layer).
    _timed(
        "file", config, lambda: store.file_entry(entry, embedder=embedder, transcript=transcript)
    )

    # Stage 8 — canonicalize against existing concepts (capped; embedding candidates first),
    # then synthesize/export the touched concept pages (design report §5).
    if config.enable_canonicalize:
        touched = _timed(
            "canonicalize",
            config,
            lambda: run_canonicalize_stage(
                entry, store, canonicalize_client or client, enable_entities=config.enable_entities
            ),
        )
        # Stage 9 — concept<->concept typed edges for the concepts just synthesized (capped;
        # centroid-similarity candidates first, Phase 16 design report §9 item 4).
        if config.enable_concept_edges:
            _timed(
                "concept_edges",
                config,
                lambda: run_concept_edges_stage(touched, store, client),
            )
    return entry


def _safe_narrative_summary(
    transcript: Transcript, summary_client: LLMClient
) -> NarrativeSummary | None:
    """Never raises: a thin-output-after-retries failure (or a dropped connection) is logged
    and leaves the entry without a narrative summary rather than failing the whole filing —
    this layer is additive, so its own failure must never regress an otherwise-successful run."""
    try:
        result = synthesize_narrative_summary(transcript.full_text(), summary_client)
    except NarrativeSummaryError:
        logger.warning("Narrative summary synthesis failed; filing without one.", exc_info=True)
        return None
    except Exception:
        logger.exception("Narrative summary synthesis raised unexpectedly; filing without one.")
        return None
    return NarrativeSummary(
        text=result.text,
        chunk_count=result.chunk_count,
        model=result.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _derive_tags(items, links, note) -> Tags:
    types = sorted({it.type for it in items})
    forms = sorted({link.application_form for link in links})
    topics = note.topics if note is not None else []
    return Tags(topics=list(topics), knowledge_types=list(types), application_forms=list(forms))


def _timed(stage: str, config: PipelineConfig, fn):
    if config.phase_callback is not None:
        config.phase_callback(stage, "start")
    start = perf_counter()
    try:
        return fn()
    finally:
        if config.timing_callback is not None:
            config.timing_callback(stage, perf_counter() - start)
        if config.phase_callback is not None:
            config.phase_callback(stage, "finish")
