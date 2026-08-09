#!/usr/bin/env python3
"""Measure the real, before-and-after token cost of adding the narrative summary layer.

Runs the real pipeline (triage, extract, link, note, and — by default — canonicalize/concepts/
entities, matching the web app's real defaults) against a representative transcript, once, with
the narrative-summary layer (`distil/summary.py`) attached via `summary_client`. Because the
narrative-summary stage is additive and never touches the strong-tier stages (see
`pipeline.run_pipeline`'s module docstring — confirmed by
`tests/unit/test_pipeline_summary.py::test_narrative_summary_runs_on_a_separate_cheap_client`),
one run gives an exact answer for both scenarios:

    before (today, no summary layer) = total cost of every strong-tier ("triage" through
                                        "canonicalize"-family) model call
    after  (with the summary layer)  = before + total cost of every cheap-tier ("summary")
                                        model call
    increase                         = the cheap-tier total, exactly

This is a real, billable run against the Anthropic API — it needs `ANTHROPIC_API_KEY` and
`DISTIL_MODEL` set (see `.env.example`). It makes no changes to the running system: everything
is filed into a throwaway temp Store, never the real `kb/`/`data/` directories.

Usage:
    ANTHROPIC_API_KEY=... DISTIL_MODEL=claude-opus-5 python scripts/measure_summary_cost.py
    # Optional: point at a real transcript instead of the bundled synthetic stand-in.
    python scripts/measure_summary_cost.py --transcript path/to/transcript.txt

No real video transcript ships in this repo (kb/okf are the owner's private data, gitignored),
so the default input is a synthetic ~45-minute talk transcript sized like a typical video this
system ingests — see `_SYNTHETIC_TRANSCRIPT` below. Pass `--transcript` to measure against a
real one instead; the relative delta (what the summary layer adds) will hold for any transcript
of similar length, but the absolute numbers are only as representative as the input.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.embed import make_embedder  # noqa: E402
from distil.ingest import ingest_file, ingest_text  # noqa: E402
from distil.llm import LLMClient  # noqa: E402
from distil.model_config import resolve_stage_model  # noqa: E402
from distil.models import LongTermGoal, Profile, StableProfile  # noqa: E402
from distil.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from distil.store import Store  # noqa: E402

# $ per 1M tokens (input, output). Source: Anthropic's published pricing, cached via the
# claude-api skill's Current Models table (2026-06-24) at the time this script was written —
# re-check platform.claude.com/docs/en/pricing before trusting this for a model not listed, or
# if prices may have changed since.
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

# A synthetic ~45-minute talk transcript, sized (not scripted) like a typical video this system
# ingests: repeated, distinct "sections" so triage/extraction see real variety rather than one
# repeated sentence. ~6,000 words / ~35,000 characters.
_SYNTHETIC_TRANSCRIPT = "\n\n".join(
    f"""Section {i}. Let me walk through another idea from this part of the talk. When you're
    building a system like this, the temptation is always to add one more layer of abstraction
    before you've actually felt the pain the abstraction is supposed to solve. I've watched teams
    spend a full sprint designing a plugin architecture for a feature that, in the end, only ever
    needed two implementations — and two implementations do not need a plugin system, they need
    an if statement. The rule I keep coming back to is: earn your abstractions. Write the second
    version of something before you generalize it, because the first version teaches you what
    actually varies and the second version teaches you what doesn't. A concrete example from a
    project I worked on: we had a notification system that started as a single email sender.
    When we added SMS, the temptation was to build a full notification-provider interface with
    retries, templating, and a plugin registry. Instead we just wrote a second function, noticed
    the two functions shared almost nothing except a recipient and a message, and only then
    extracted the two lines that were actually common. That thin extraction survived three more
    channels without a single rewrite, because it was earned from real duplication rather than
    guessed at in advance. The broader point — and this is the part people skip — is that
    premature abstraction isn't just wasted effort, it actively costs you later: every generic
    interface has to be understood by the next person before they can change anything, and a
    wrong generalization is far harder to undo than a late one is to add.
    """
    for i in range(1, 13)
)


@dataclass
class _Usage:
    model: str
    input_tokens: int
    output_tokens: int


class _TrackingClient(LLMClient):
    """Real Anthropic calls, with usage recorded per call — the measurement instrument itself.
    Not `distil.llm.AnthropicClient`, which discards `usage` after extracting the text."""

    def __init__(self, model: str):
        self.model = model
        self.usages: list[_Usage] = []

    def _record_and_extract(self, message) -> str:
        self.usages.append(
            _Usage(self.model, message.usage.input_tokens, message.usage.output_tokens)
        )
        return "".join(block.text for block in message.content if block.type == "text")

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict = {
            "model": self.model, "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return self._record_and_extract(client.messages.create(**kwargs))

    def stream(self, prompt: str, *, system: str | None = None):  # pragma: no cover - unused
        text = self.complete(prompt, system=system)
        words = text.split(" ")
        for i, w in enumerate(words):
            yield (w if i == 0 else " " + w)


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


def _report(label: str, usages: list[_Usage]) -> float:
    cost = _cost(usages)
    input_tok = sum(u.input_tokens for u in usages)
    output_tok = sum(u.output_tokens for u in usages)
    print(f"{label}: {len(usages)} call(s), {input_tok} input tok, {output_tok} output tok, "
          f"${cost:.4f}")
    return cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, default=None,
                        help="Path to a real .srt/.txt/.md transcript (default: bundled synthetic)")
    args = parser.parse_args()

    transcript = ingest_file(args.transcript) if args.transcript else ingest_text(_SYNTHETIC_TRANSCRIPT)

    strong_model = resolve_stage_model("triage")
    summary_model = resolve_stage_model("summary")
    print(f"Strong-tier model (triage/extract/link/note/canonicalize): {strong_model}")
    print(f"Cheap-tier model (summary): {summary_model}")
    print()

    strong_client = _TrackingClient(strong_model)
    summary_client = _TrackingClient(summary_model)

    profile = Profile(
        user_id="owner",
        stable=StableProfile(long_term_goals=[
            LongTermGoal(id="g_01", statement="Write maintainable software",
                         created_at="2026-01-01T00:00:00")
        ]),
    )

    try:
        embedder = make_embedder()
    except Exception:
        embedder = None

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(db_path=Path(tmp) / "d.db", kb_dir=Path(tmp) / "kb")
        run_pipeline(
            transcript, profile, store, strong_client,
            source_title="Representative talk",
            config=PipelineConfig(enable_graph=False),
            embedder=embedder,
            summary_client=summary_client,
        )

    print()
    before = _report("Before (today, no narrative summary)", strong_client.usages)
    summary_cost = _report("Narrative summary layer only", summary_client.usages)
    after = before + summary_cost
    print()
    print(f"After (with narrative summary):        ${after:.4f}")
    print(f"Increase:                               ${after - before:.4f} "
          f"({(after / before - 1) * 100:.1f}% more than before)" if before else "")


if __name__ == "__main__":
    main()
