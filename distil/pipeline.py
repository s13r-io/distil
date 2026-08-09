"""Pipeline orchestration — wires stages 0→9. ARCHITECTURE.md §2; TESTING T-PL1, T-PL2, T-PL5, T-PL6.

One call turns a normalized transcript + profile into a filed, schema-valid :class:`KBEntry`:

    ingest (done by caller) → triage → chunked extract → normalize → link
    → note synthesis → graph → file → canonicalize → concept edges
                        (narrative summary starts at the front, runs concurrently, joins before filing)

**There is no quality short-circuit here.** The owner chose the video (or note); that editorial
judgment is trusted unconditionally, and a model second-guessing it can only subtract. The one
rejection rule — a transcript below a word-count floor — is enforced earlier, at ingest
(``ingest.py``'s ``TranscriptTooShortError``), never here: by the time a ``Transcript`` reaches
this function, it is always worth filing, and every entry that starts this pipeline finishes it
filed (Stage 7 always runs; T-PL2). Triage's ``verdict`` is still computed and stored (other code
and stored data depend on its shape) but nothing here acts on it — only
``knowledge_types_present`` (routes what ``extract.run_chunked_extraction`` looks for),
``density``, and ``transcript_loss`` (informational) matter downstream.

**Triage and extraction are two separate calls, on two different tiers (owner decision).** They
were briefly merged into one strong-tier call; that merge is undone here because it is
incompatible with chunked extraction (several per-chunk calls cannot produce one
whole-transcript classification without disagreeing verdicts — see ``extract.py``'s module
docstring). ``triage.run_triage`` classifies the whole transcript once, on the cheap tier (a
coarse categorical judgement, not the faithfulness-critical work); its dominant type then steers
``extract.run_chunked_extraction``, on the strong tier, preserving decide-then-act across the two
calls via that data dependency. Chunking a long transcript instead of extracting it in one
whole-transcript call is itself measured to find materially more faithful items — see
``extract.py``'s module docstring for the measurement and how boundary damage/duplication are
handled. The LLM-call budget scales with chunk count for extraction specifically (one call per
chunk instead of one call for the whole transcript) but stays otherwise bounded (one triage call
+ link + note, plus capped graph calls and capped canonicalize/synthesis/concept-edge calls — see
``canonicalize.py``'s and ``concept_graph.py``'s module docstrings and the OKF Phase 3 design
report §6, §9 item 4).

**The narrative summary runs concurrently with triage+extract, not before or after them**
(``distil/summary.py``). It is additive and optional: it only runs when a caller passes
``summary_client`` (a *separate*, cheap-tier client — never ``client``, which stays on the
strong model for extract/canonicalize unless overridden) and ``config.enable_narrative_summary``
is true. It shares no data with triage/extract (all three read only the transcript), so there is
no ordering dependency to preserve by blocking — starting it at the front means the owner sees a
readable account sooner, and finishing early on a cheap tier means it survives a later extraction
failure instead of dying behind it. It runs on a plain daemon ``threading.Thread`` (stdlib, no
new dependency) started right before triage is dispatched, and is joined — bounded by
``DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS`` (default 60s) — only once the rest of the pipeline
(triage, extract, normalize, link, note) has already produced ``entry``, so the common case pays
no extra wall-clock time at all. A summary that fails (a thin-output-after-retries failure, or a
dropped connection) or that is still running past the bound both leave ``entry.narrative_summary``
``None`` — logged either way (never silently) — rather than ever blocking filing or turning a
would-have-succeeded run into a failed one; the owner's existing refresh action can always
generate it later. A failure or timeout in the summary never masks, delays, or is masked by an
extraction failure: they are independent stages evaluated on separate threads, and either one's
outcome is reported on its own terms. The thread is deliberately plain and daemonic rather than a
``concurrent.futures.ThreadPoolExecutor``: that class registers an ``atexit`` hook that blocks
interpreter shutdown until every worker thread finishes, even after ``shutdown(wait=False)`` —
which would silently defeat the whole point of the bound for a one-shot CLI run (`distil run`
would hang at process exit waiting for an abandoned, already-timed-out-and-ignored summary
call). A daemon thread is simply killed when the process exits, with no such wait.

Per-stage model selection (``distil/model_config.py``) is a general mechanism, not something
built one-off for the narrative summary: ``triage_client``/``extract_client``/``link_client``/
``note_client``/``graph_client``/``canonicalize_client`` let a caller inject a distinct client
per stage, each defaulting to ``None`` — meaning "use ``client``". ``cli.py`` and ``web/app.py``
both construct all six via ``model_config.make_stage_client(stage)`` (each stage's own
``_make_<stage>_client`` seam) and pass them through these kwargs, so setting
``DISTIL_MODEL_<STAGE>`` in the environment genuinely changes only that stage's model — no
further ``pipeline.py`` change required, and no caller needs to pass these explicitly to get
today's defaults (every stage still resolves to its tier default unless overridden — see
``model_config.py``'s ``STRONG_TIER_STAGES``/``CHEAP_TIER_STAGES``). A settings UI for editing
these values is a deliberate follow-up; the wiring itself is not.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter

from .canonicalize import run_canonicalize_stage
from .concept_graph import run_concept_edges_stage
from .embed import Embedder
from .extract import run_chunked_extraction
from .graph import link_graph
from .ingest import Transcript
from .link import generate_links
from .llm import LLMClient
from .models import EntryMeta, KBEntry, NarrativeSummary, Profile, Source, Tags
from .normalize import normalize_items
from .note import synthesize_note
from .store import Store
from .summary import NarrativeSummaryError, synthesize_narrative_summary
from .triage import run_triage

logger = logging.getLogger(__name__)

_DEFAULT_SUMMARY_JOIN_TIMEOUT_SECONDS = 60.0


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
    # Reports stage start/finish for live progress display, independent of timing_callback's
    # after-the-fact duration reporting. Events: ("<stage>", "start"), ("<stage>", "finish").
    # narrative_summary's "start" fires when the background thread is kicked off (the front of
    # the pipeline) and its "finish" fires only once it is joined (after note) — the two events
    # for that one stage are the exception to "no stage overlaps another": everything else here
    # remains strictly sequential from this callback's point of view, since it is only ever
    # invoked from the main thread, never from the background summary thread.
    # (Callers may also feed this same reporter a ("<stage>", "short_circuit") event of their
    # own for a rejection that happens outside this function entirely — see web/app.py's
    # TranscriptTooShortError handling — but run_pipeline itself never emits one.)
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
        transcript_word_count=len(transcript.full_text().split()),
        captured_at=now,
    )
    meta = EntryMeta(created_at=now, model_version=config.model_version)

    # Narrative summary starts at the FRONT, concurrently with triage+extraction below — see
    # module docstring. Never blocks anything here; joined only once entry exists.
    summary_run: _SummaryRun | None = None
    summary_thread: threading.Thread | None = None
    summary_started_at = 0.0
    if config.enable_narrative_summary and summary_client is not None:
        if config.phase_callback is not None:
            config.phase_callback("narrative_summary", "start")
        summary_started_at = perf_counter()
        summary_run = _SummaryRun()
        summary_thread = threading.Thread(
            target=summary_run.execute,
            args=(transcript, summary_client),
            daemon=True,
        )
        summary_thread.start()

    # Stage 1 — triage (cheap tier): classifies the whole transcript once.
    triage = _timed(
        "triage", config, lambda: run_triage(transcript, triage_client or client).triage
    )

    # Stage 2 — extraction (strong tier), chunked: see module docstring and
    # extract.run_chunked_extraction's own docstring for why.
    extraction_result = _timed(
        "extract",
        config,
        lambda: run_chunked_extraction(transcript, triage, extract_client or client),
    )
    raw_items = extraction_result.items
    extraction_truncated = extraction_result.truncated

    # Stage 3 — normalize (pure faithfulness gate).
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
        extraction_truncated=extraction_truncated,
    )

    # Join the narrative summary now that the rest of the pipeline has already done its work —
    # in the common case it finished long ago and this costs no extra wall-clock time. Bounded:
    # it must never delay filing indefinitely (owner decision). Either outcome — a genuine
    # failure inside the stage, or still running past the bound — is logged honestly and leaves
    # narrative_summary unset; the entry always still files (see module docstring).
    if summary_run is not None:
        assert summary_thread is not None
        entry.narrative_summary = _join_narrative_summary(
            summary_thread, summary_run, summary_started_at, config
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


@dataclass
class _SummaryRun:
    """Holds the background narrative-summary thread's result. A plain mutable container rather
    than a ``concurrent.futures.Future`` — see the module docstring for why a plain daemon
    thread was chosen over ``ThreadPoolExecutor`` here."""

    result: NarrativeSummary | None = field(default=None)

    def execute(self, transcript: Transcript, summary_client: LLMClient) -> None:
        """Runs on the background thread — must not touch anything besides its own arguments
        and ``self.result``. Never raises: a thin-output-after-retries failure (or a dropped
        connection) is logged and leaves ``result`` unset rather than propagating — this layer
        is additive, so its own failure must never regress an otherwise-successful run."""
        try:
            summary = synthesize_narrative_summary(transcript.full_text(), summary_client)
        except NarrativeSummaryError:
            logger.warning(
                "Narrative summary synthesis failed; filing without one.", exc_info=True
            )
            return
        except Exception:
            logger.exception(
                "Narrative summary synthesis raised unexpectedly; filing without one."
            )
            return
        self.result = NarrativeSummary(
            text=summary.text,
            chunk_count=summary.chunk_count,
            model=summary.model,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


def _join_narrative_summary(
    thread: threading.Thread, run: _SummaryRun, started_at: float, config: PipelineConfig
) -> NarrativeSummary | None:
    """Wait for the background narrative-summary thread, bounded by
    ``DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS`` — it must never delay filing indefinitely. A timeout
    is logged just as honestly as an in-stage failure (``_SummaryRun.execute`` already logs its
    own); the thread is left running to completion regardless (it's a daemon — the process can
    still exit without waiting for it), but nothing ever reads ``run.result`` again after this
    point.
    """
    timeout = _summary_join_timeout_seconds()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            "Narrative summary still running after %.0fs (DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS); "
            "filing without one — it can be generated later via the refresh action.",
            timeout,
        )
        result = None
    else:
        result = run.result
    if config.timing_callback is not None:
        config.timing_callback("narrative_summary", perf_counter() - started_at)
    if config.phase_callback is not None:
        config.phase_callback("narrative_summary", "finish")
    return result


def _summary_join_timeout_seconds() -> float:
    try:
        return float(
            os.environ.get(
                "DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS", _DEFAULT_SUMMARY_JOIN_TIMEOUT_SECONDS
            )
        )
    except ValueError:
        return _DEFAULT_SUMMARY_JOIN_TIMEOUT_SECONDS


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
