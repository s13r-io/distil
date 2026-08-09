#!/usr/bin/env python3
"""Measure the real before/after effect of the per-stage max_tokens ceiling fix on extraction.

`AnthropicClient.complete()` used to hardcode `max_tokens=4096` for every stage — extraction
included, even though extraction's output scales with transcript length (one JSON object per
knowledge item — see `distil/extract.py`'s module docstring). On a real ~1,455-word transcript,
extraction hit that ceiling *exactly* and the existing salvage path silently recovered a partial
item list with no error, no warning, and an entry that filed and looked complete.
`distil/model_config.py` now resolves extraction's ceiling to the model's own published output
limit instead (`resolve_stage_max_tokens`).

This script runs one cheap-tier `distil.triage.run_triage` call to get a dominant type, then the
real extraction call(s) (`distil.extract.run_extraction`/`run_chunked_extraction`) conditioned on
that type — not the full pipeline, since link/note/graph/file are irrelevant to this defect and
would only add cost/noise — against a real transcript you point it at, three ways:

    1. BEFORE  — the old hardcoded max_tokens=4096, exactly as production behaved until this fix.
    2. AFTER   — the new resolved ceiling (`model_config.resolve_stage_max_tokens("extract")`).
    3. CHUNKED — the transcript split via `distil.summary.chunk_transcript_text` (reused, not
                 reimplemented — the same sentence-safe chunker the narrative summary layer
                 already uses), one `run_extraction`-equivalent call per chunk at the new ceiling
                 (`distil.extract.run_chunked_extraction`).

For each, it reports item count, whether the response was truncated/salvaged
(`ChunkedExtractionResult.truncated`/the single-call path's own truncation flag — the same signal
recorded on `KBEntry.extraction_truncated`), and the real dollar cost from the API's own reported
token usage.

This is a real, billable run against the Anthropic API — it needs `ANTHROPIC_API_KEY` and
`DISTIL_MODEL` set (see `.env.example`). It makes no changes to any Store — it never files
anything, touches no `kb/`/`data/` directory, real or throwaway.

Usage:
    ANTHROPIC_API_KEY=... DISTIL_MODEL=claude-sonnet-5 python scripts/measure_extraction_ceiling.py \\
        --transcript /path/to/transcript.txt [--transcript /path/to/another.txt]

No transcript is bundled with this script and none should ever be added here — always pass a
real transcript's path on disk via `--transcript`; this script only reads it, never copies or
commits it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.extract import dominant_type, run_chunked_extraction, run_extraction  # noqa: E402
from distil.ingest import Transcript, ingest_file  # noqa: E402
from distil.llm import LLMClient  # noqa: E402
from distil.model_config import resolve_stage_max_tokens, resolve_stage_model  # noqa: E402
from distil.models import Triage  # noqa: E402
from distil.triage import run_triage  # noqa: E402

# $ per 1M tokens (input, output). Source: Anthropic's published pricing, cached via the
# claude-api skill's Current Models table at the time this script was written — re-check
# platform.claude.com/docs/en/pricing before trusting this for a model not listed, or if prices
# may have changed since.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# The literal this whole task exists to replace (distil/llm.py:119 before the fix).
_OLD_HARDCODED_MAX_TOKENS = 4096


@dataclass
class _Usage:
    model: str
    input_tokens: int
    output_tokens: int


class _TrackingClient(LLMClient):
    """Real Anthropic calls at a configurable max_tokens, with usage recorded per call. Not
    `distil.llm.AnthropicClient`, which discards `usage` after extracting the text — this is the
    measurement instrument itself. Always streams (regardless of max_tokens) so one client works
    uniformly for both the old 4096 ceiling and the new, much larger one — see llm.py's own
    streaming-threshold comment for why a large non-streaming call is unsafe."""

    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.usages: list[_Usage] = []

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
        self.usages.append(
            _Usage(self.model, message.usage.input_tokens, message.usage.output_tokens)
        )
        return "".join(block.text for block in message.content if block.type == "text")

    def stream(self, prompt: str, *, system: str | None = None):  # pragma: no cover - unused
        text = self.complete(prompt, system=system)
        yield text


def _cost(usages: list[_Usage]) -> float:
    total = 0.0
    for u in usages:
        if u.model not in _PRICING_PER_MTOK:
            raise SystemExit(
                f"No pricing entry for model '{u.model}' — add it to _PRICING_PER_MTOK before "
                "trusting this script's output for that model."
            )
        rate_in, rate_out = _PRICING_PER_MTOK[u.model]
        total += u.input_tokens / 1_000_000 * rate_in + u.output_tokens / 1_000_000 * rate_out
    return total


class _TruncationWatcher(logging.Handler):
    """Detects whether `run_extraction` logged its truncation warning during a call — the
    single-call path only logs truncation (see `extract.py`), it doesn't return the flag the
    way `ChunkedExtractionResult.truncated` does."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.triggered = False

    def emit(self, record: logging.LogRecord) -> None:
        self.triggered = True


def _run_single_call(
    transcript: Transcript, triage: Triage, model: str, max_tokens: int, label: str
) -> None:
    client = _TrackingClient(model, max_tokens)
    watcher = _TruncationWatcher()
    extract_logger = logging.getLogger("distil.extract")
    extract_logger.addHandler(watcher)
    try:
        items = run_extraction(transcript, triage, client)
    finally:
        extract_logger.removeHandler(watcher)
    cost = _cost(client.usages)
    in_tok = sum(u.input_tokens for u in client.usages)
    out_tok = sum(u.output_tokens for u in client.usages)
    print(
        f"{label}: max_tokens={max_tokens} -> {len(items)} item(s), "
        f"truncated={watcher.triggered}, {len(client.usages)} call(s), "
        f"{in_tok} in / {out_tok} out tok, ${cost:.4f}"
    )
    if watcher.triggered:
        print("  ^ truncated=True: the response was salvaged from an incomplete array — this "
              "run silently lost knowledge before the ceiling fix would have caught it.")
    print("  statements:")
    for item in items:
        print(f"    [{item.type}] {item.statement}")


def _run_chunked(transcript: Transcript, triage: Triage, model: str, max_tokens: int) -> None:
    client = _TrackingClient(model, max_tokens)
    result = run_chunked_extraction(transcript, triage, client)
    cost = _cost(client.usages)
    in_tok = sum(u.input_tokens for u in client.usages)
    out_tok = sum(u.output_tokens for u in client.usages)
    print(
        f"chunked: {result.chunk_count} chunk(s) -> {len(result.items)} item(s) total, "
        f"any_truncated={result.truncated}, {len(client.usages)} call(s), "
        f"{in_tok} in / {out_tok} out tok, ${cost:.4f}"
    )
    print("  statements:")
    for item in result.items:
        print(f"    [{item.type}] {item.statement}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", type=Path, action="append", required=True,
                        help="Path to a real .srt/.txt/.md transcript. Repeatable.")
    args = parser.parse_args()

    model = resolve_stage_model("extract")
    triage_model = resolve_stage_model("triage")
    triage_max_tokens = resolve_stage_max_tokens("triage")
    new_ceiling = resolve_stage_max_tokens("extract")
    print(f"Model: {model}")
    print(f"Old hardcoded ceiling (before this fix): {_OLD_HARDCODED_MAX_TOKENS}")
    print(f"New resolved ceiling (after this fix):   {new_ceiling}")
    print()

    for path in args.transcript:
        transcript = ingest_file(path)
        word_count = len(transcript.full_text().split())
        print(f"=== {path} ({word_count} words as ingested, {len(transcript.segments)} "
              f"segment(s)) ===")
        print()

        triage_client = _TrackingClient(triage_model, triage_max_tokens)
        triage = run_triage(transcript, triage_client).triage
        print(f"Triage (cheap tier, runs once): dominant type = {dominant_type(triage)}")
        print()

        print("-- BEFORE (old hardcoded max_tokens=4096) --")
        _run_single_call(transcript, triage, model, _OLD_HARDCODED_MAX_TOKENS, "single-call")
        print()
        print("-- AFTER (new resolved ceiling) --")
        _run_single_call(transcript, triage, model, new_ceiling, "single-call")
        print()
        print("-- CHUNKED (new ceiling per chunk, distil.summary.chunk_transcript_text) --")
        _run_chunked(transcript, triage, model, new_ceiling)
        print()


if __name__ == "__main__":
    main()
