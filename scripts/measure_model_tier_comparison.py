#!/usr/bin/env python3
"""Compare the full pipeline's real output/cost across three model-tier configurations.

Runs the real, full pipeline (extract, normalize, link, note, narrative summary, canonicalize +
entities — graph/concept-edges disabled, see below) against one real transcript, three times:

    1. all-sonnet — every stage on `claude-sonnet-5`.
    2. all-haiku  — every stage on `claude-haiku-4-5`.
    3. split      — extract + canonicalize on `claude-sonnet-5`; link, note, and narrative
                    summary on `claude-haiku-4-5`.

**Triage cannot be independently assigned a model in the split.** Triage and extraction are
merged into one call (owner decision — see `distil/extract.py`'s and `distil/pipeline.py`'s
module docstrings; this repo does not reintroduce a separate triage call), so "triage" always
runs on whatever model `extract` uses. A request for "triage on the cheap tier, extraction on
the strong tier" is not achievable without splitting that merged call back apart, which is out
of scope here (and explicitly not something this codebase does anymore). This script reports
that plainly rather than silently reinterpreting the request.

Graph linking and concept-edge computation are disabled (`enable_graph=False`,
`enable_concept_edges=False`): each configuration runs against its own fresh, empty Store, so
`link_graph`'s deterministic candidate lookup always finds zero candidates (no other entries
exist to link against) — leaving them on would cost nothing but add noise to the report.
Canonicalize (concept + entity decisions) is left on, since that's one of the things this
comparison is meant to judge.

For each configuration, this script:
  - reports per-stage call count, token usage, and real dollar cost;
  - reports the item count before and after the faithfulness gate (`normalize.py`), and flags
    any item whose `provenance.quote` fails the deterministic verbatim-match check
    (`distil.faithfulness.quote_in_transcript`) *before* normalize drops it — this is the
    objective, programmatic reading of "quote fidelity," not a subjective one;
  - reports concept/entity counts (a rough, single-video proxy for "concept merge decisions" —
    a real merge-vs-new judgment needs a second video in the store, which this comparison
    doesn't have; see the printed caveat);
  - files the entry into a persistent (not deleted) directory so a human can read the full
    kb/<id>.md, okf/concepts/*.md, and okf/entities/*.md pages side by side across the three runs.

This is a real, billable run against the Anthropic API — it needs `ANTHROPIC_API_KEY` set (see
`.env.example`). `DISTIL_MODEL` is not read; the two model IDs are passed explicitly.

Usage:
    ANTHROPIC_API_KEY=... python scripts/measure_model_tier_comparison.py \\
        --transcript /path/to/transcript.txt --out-dir /path/to/writable/dir

No transcript is bundled with this script — always pass a real one via `--transcript`; this
script only reads it, never copies or commits it. `--out-dir` must be writable and is never
inside this repository — point it somewhere throwaway (e.g. a scratch directory).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.embed import make_embedder  # noqa: E402
from distil.faithfulness import quote_in_transcript  # noqa: E402
from distil.ingest import Segment, Transcript, ingest_file  # noqa: E402
from distil.llm import LLMClient  # noqa: E402
from distil.models import LongTermGoal, Profile, StableProfile  # noqa: E402
from distil.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from distil.store import Store  # noqa: E402
from distil import pipeline as pipeline_mod  # noqa: E402

_SONNET = "claude-sonnet-5"
_HAIKU = "claude-haiku-4-5"

# Mirrors model_config.py's MODEL_MAX_OUTPUT_TOKENS / _DEFAULT_STAGE_MAX_TOKENS *by value*,
# not by importing the real functions: model_config.resolve_stage_max_tokens("extract")
# re-derives the model from DISTIL_MODEL/DISTIL_MODEL_EXTRACT in the environment, which this
# script cannot use — it needs to vary the model independently of env to run three
# configurations in one process. Keep these two tables in sync with model_config.py by hand.
_MODEL_MAX_OUTPUT_TOKENS = {_SONNET: 128_000, _HAIKU: 64_000}
_DEFAULT_STAGE_MAX_TOKENS = 4096


def _max_tokens_for(model: str, stage: str) -> int:
    if stage != "extract":
        return _DEFAULT_STAGE_MAX_TOKENS
    return _MODEL_MAX_OUTPUT_TOKENS.get(model, _DEFAULT_STAGE_MAX_TOKENS)


# ---- MM:SS-per-line caption ingest workaround (measurement-only; does not touch ingest.py) ----
#
# distil/ingest.py's _INLINE_TS only matches HH:MM:SS timestamps, not the MM:SS-per-line shape
# some real caption exports use (confirmed on the owner's hourlong.txt: 1-2 digit minutes, colon,
# 2-digit seconds, e.g. "05:13 designing the environment..."). ingest_file() therefore falls
# through to _parse_paragraphs, which does not recognize or strip these timestamps at all — every
# segment's text ends up with stray "MM SS" digit tokens spliced into the middle of what was
# continuous prose, and empirically this breaks distil.faithfulness.quote_in_transcript almost
# completely (verified: a full-pipeline run against the real, unmodified ingest_file() output
# dropped 17/17 extracted items as "unfaithful" — every one was a real quote broken up by
# injected timestamp tokens, not a model hallucination). This is a real, separate, pre-existing
# ingest defect — flagged plainly in the PR, and NOT fixed here, since fixing it is out of this
# task's scope. This function exists solely so the model-tier comparison below measures actual
# model behavior instead of being swamped by an unrelated ingest bug; it reproduces what a fixed
# _INLINE_TS/_parse_inline_timestamped would produce for this one file shape.
_MM_SS_LINE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s+(.*)$")


def _ingest_mm_ss_captions(raw_text: str) -> Transcript:
    segments: list[Segment] = []
    idx = 0
    for line in raw_text.splitlines():
        m = _MM_SS_LINE.match(line)
        if not m:
            continue
        body = m.group(3).strip()
        if not body:
            continue
        segments.append(Segment(text=body, locator=f"seg:{idx}", timestamp=f"{m.group(1)}:{m.group(2)}"))
        idx += 1
    if not segments:
        raise ValueError("No MM:SS-prefixed lines with body text found — wrong file shape?")
    return Transcript(segments=segments)

# $ per 1M tokens (input, output). Source: Anthropic's published pricing, cached via the
# claude-api skill's Current Models table at the time this script was written.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_CONFIGS: dict[str, dict[str, str]] = {
    "all-sonnet": {
        "extract": _SONNET, "link": _SONNET, "note": _SONNET,
        "canonicalize": _SONNET, "summary": _SONNET,
    },
    "all-haiku": {
        "extract": _HAIKU, "link": _HAIKU, "note": _HAIKU,
        "canonicalize": _HAIKU, "summary": _HAIKU,
    },
    "split": {
        "extract": _SONNET, "canonicalize": _SONNET,
        "link": _HAIKU, "note": _HAIKU, "summary": _HAIKU,
    },
}


@dataclass
class _Usage:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class _RunStats:
    usages: list[_Usage] = field(default_factory=list)
    raw_item_count: int = 0
    normalized_item_count: int = 0
    unfaithful_raw_quotes: list[str] = field(default_factory=list)


class _TrackingClient(LLMClient):
    """Real Anthropic calls for one stage, at that stage's real resolved max_tokens ceiling
    (`model_config.resolve_stage_max_tokens`), with usage recorded per call by stage name."""

    def __init__(self, model: str, stage: str, stats: _RunStats):
        self.model = model
        self.max_tokens = _max_tokens_for(model, stage)
        self._stage = stage
        self._stats = stats

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict = {
            "model": self.model, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        self._stats.usages.append(
            _Usage(self._stage, self.model, message.usage.input_tokens, message.usage.output_tokens)
        )
        return "".join(block.text for block in message.content if block.type == "text")

    def stream(self, prompt: str, *, system: str | None = None):  # pragma: no cover - unused
        yield self.complete(prompt, system=system)


def _cost(usages: list[_Usage]) -> float:
    total = 0.0
    for u in usages:
        rate_in, rate_out = _PRICING_PER_MTOK[u.model]
        total += u.input_tokens / 1_000_000 * rate_in + u.output_tokens / 1_000_000 * rate_out
    return total


def _cost_by_stage(usages: list[_Usage]) -> dict[str, float]:
    by_stage: dict[str, list[_Usage]] = {}
    for u in usages:
        by_stage.setdefault(u.stage, []).append(u)
    return {stage: _cost(us) for stage, us in by_stage.items()}


def _run_one(
    config_name: str, models: dict[str, str], transcript: Transcript, out_root: Path,
) -> None:
    stats = _RunStats()
    clients = {stage: _TrackingClient(model, stage, stats) for stage, model in models.items()}

    # Wrap normalize_items (as bound in pipeline.py's own namespace) to record the raw vs.
    # post-faithfulness-gate item count, and independently check each raw item's quote against
    # the transcript before normalize ever drops it — the objective "quote fidelity" signal.
    original_normalize = pipeline_mod.normalize_items

    def _tracking_normalize(items, transcript_arg):
        stats.raw_item_count = len(items)
        stats.unfaithful_raw_quotes = [
            it.provenance.quote for it in items if not quote_in_transcript(it.provenance.quote, transcript_arg)
        ]
        result = original_normalize(items, transcript_arg)
        stats.normalized_item_count = len(result)
        return result

    pipeline_mod.normalize_items = _tracking_normalize
    try:
        profile = Profile(
            user_id="owner",
            stable=StableProfile(long_term_goals=[
                LongTermGoal(id="g_01", statement="Understand AI agent architecture",
                             created_at="2026-01-01T00:00:00")
            ]),
        )
        try:
            embedder = make_embedder()
        except Exception:
            embedder = None

        run_dir = out_root / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(db_path=run_dir / "distil.db", kb_dir=run_dir / "kb")

        entry = run_pipeline(
            transcript, profile, store, clients["link"],
            source_title="Model-tier comparison transcript",
            config=PipelineConfig(enable_graph=False, enable_concept_edges=False),
            embedder=embedder,
            summary_client=clients["summary"],
            extract_client=clients["extract"],
            link_client=clients["link"],
            note_client=clients["note"],
            canonicalize_client=clients["canonicalize"],
        )
    finally:
        pipeline_mod.normalize_items = original_normalize

    concepts = store.list_concepts()
    entities = store.list_entities()
    total_cost = _cost(stats.usages)

    print(f"=== {config_name} ({', '.join(f'{k}={v}' for k, v in models.items())}) ===")
    print(f"  filed: {store.entry_path(entry.entry_id)}")
    print(f"  items: {stats.raw_item_count} raw -> {stats.normalized_item_count} after "
          f"faithfulness gate (dropped {stats.raw_item_count - stats.normalized_item_count})")
    if stats.unfaithful_raw_quotes:
        print(f"  UNFAITHFUL raw quotes (would fail verbatim match, dropped by normalize):")
        for q in stats.unfaithful_raw_quotes:
            print(f"    - {q!r}")
    else:
        print("  quote fidelity: every raw item's quote verbatim-matched the transcript")
    print(f"  extraction_truncated: {entry.extraction_truncated}")
    print(f"  narrative_summary: {'present' if entry.narrative_summary else 'absent'}")
    print(f"  concepts created/touched: {len(concepts)}")
    for c in concepts:
        print(f"    - {c.concept_id}: {c.title} ({len(c.members)} member(s))")
    print(f"  entities created/touched: {len(entities)}")
    for e in entities:
        print(f"    - {e.entity_id} [{e.kind}]: {e.title} ({len(e.members)} member(s))")
    print(f"  cost by stage: " + ", ".join(
        f"{stage}=${c:.4f}" for stage, c in sorted(_cost_by_stage(stats.usages).items())
    ))
    print(f"  TOTAL COST: ${total_cost:.4f}  ({len(stats.usages)} call(s))")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", type=Path, required=True,
                        help="Path to a real .srt/.txt/.md transcript.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Writable directory (outside this repo) to file each run's "
                             "throwaway kb/store into, one subdirectory per configuration.")
    parser.add_argument("--mm-ss-captions", action="store_true",
                        help="This transcript uses MM:SS-per-line timestamps that "
                             "distil.ingest.ingest_file() does not parse correctly today (see "
                             "the module comment above _ingest_mm_ss_captions) — use the "
                             "measurement-only workaround parser instead of the real ingest "
                             "path. Only use this after confirming the file actually has that "
                             "shape; it is not a general-purpose ingest replacement.")
    args = parser.parse_args()

    if args.mm_ss_captions:
        print("*** WARNING: using the MM:SS-captions ingest WORKAROUND, not the real "
              "distil.ingest.ingest_file() path — see this script's module docstring and the "
              "_ingest_mm_ss_captions comment. This is a measurement-only stand-in for a real, "
              "unfixed ingest.py defect; production behavior on this file differs. ***")
        transcript = _ingest_mm_ss_captions(args.transcript.read_text(encoding="utf-8"))
    else:
        transcript = ingest_file(args.transcript)
    word_count = len(transcript.full_text().split())
    print(f"Transcript: {args.transcript} ({word_count} words as ingested)")
    print(f"Output root: {args.out_dir}")
    print()

    for config_name, models in _CONFIGS.items():
        if config_name == "split":
            print("NOTE: triage is merged into the same call as extract (owner decision — see "
                  "distil/extract.py's module docstring), so triage always runs on whatever "
                  "model 'extract' uses in this config (sonnet), not the haiku tier this split "
                  "was asked to put it on. That specific request isn't achievable without "
                  "un-merging triage from extraction, which is out of scope.")
        _run_one(config_name, models, transcript, args.out_dir)


if __name__ == "__main__":
    main()
