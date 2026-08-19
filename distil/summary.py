"""Narrative summary layer — reads the whole transcript, unlike ``note.py``.

``note.py`` builds the reader-facing teaching note from ``KnowledgeItem`` objects only — the
items already survived triage/extraction's filtering. That filtering is exactly what discards
the argument's build-up, worked examples, and digressions that never became a discrete
quotable claim, so no amount of prompt-tuning on the note synthesis step can recover them: the
note never sees the transcript. This module fixes that by summarizing the transcript directly,
chunk by chunk (so a single completion never has to compress an entire long video, and so a
too-thin chunk summary is a rejectable failure rather than a shrug), then merging the chunk
summaries into one flowing account.

The final merged account, never each chunk, may then pass through ``unslop.py``'s two-call
rewrite/self-audit on the same summary-tier client. Its output must still satisfy the merge
coverage floor; failure keeps the already-valid pre-unslop merge.

This is a strictly additive companion to the grounded note, not a replacement — see
``models.NarrativeSummary``'s docstring and the module docstring in ``distil/note.py``. It also
runs on a **cheaper model tier** than extraction/note/concepts/entities: summarizing spoken
text into prose is compression, not judgement, so the caller is expected to inject a client
built from the ``"summary"`` stage's model (see ``distil/model_config.py``'s per-stage
resolution mechanism and ``cli._make_summary_client``), never the strong-tier client used for
every other pipeline stage.

Coverage guard: a chunk (or the merged whole) that comes back far too short for the material it
covered is a failure, not a result — ``_min_chunk_summary_len``/``_min_merge_summary_len`` scale
a minimum length against the source length, and both retry a bounded number of times
(``DISTIL_SUMMARY_MAX_RETRIES``) before raising :class:`NarrativeSummaryError`. Callers decide
what "fail" means for them: ``pipeline.py``'s stage catches it and simply leaves
``narrative_summary`` unset (additive — never blocks filing), while a manual refresh
(``refresh_summary.py``) surfaces it to the caller as an honest error message.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from .llm import LLMClient
from .model_config import resolve_stage_model
from .prompts.summary import (
    SYSTEM_CHUNK,
    SYSTEM_MERGE,
    build_chunk_prompt,
    build_merge_prompt,
)
from .unslop import rewrite_text

logger = logging.getLogger(__name__)

# ~2,000-2,500 words (~15-20 minutes of spoken content at typical speaking pace): small enough
# that a single completion summarizes a chunk thoroughly without truncating or falling back to
# a one-sentence gloss, large enough that a typical 30-90 minute video needs only a handful of
# chunks (and API calls) rather than dozens.
DEFAULT_CHUNK_CHARS = 12_000

DEFAULT_MAX_RETRIES = 3

# A chunk summary must be at least this fraction of its source chunk's character length, with
# an absolute floor for very short chunks/tails. This is the guard the task calls for: a long
# passage dismissed in a sentence is rejected and retried, not accepted.
MIN_CHUNK_SUMMARY_RATIO = 0.08
MIN_CHUNK_SUMMARY_FLOOR = 40

# The merge is allowed to condense (it removes redundancy the chunking itself introduced at
# chunk boundaries), but not to re-flatten the whole video down to a paragraph — it must retain
# a majority of the combined chunk-summary substance.
MIN_MERGE_SUMMARY_RATIO = 0.5

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class NarrativeSummaryError(ValueError):
    """Raised when a narrative summary can't be produced within the retry budget."""


def chunk_transcript_text(text: str, chunk_chars: int | None = None) -> list[str]:
    """Split ``text`` into chunks of roughly ``chunk_chars`` characters, never splitting a
    sentence. A single sentence longer than ``chunk_chars`` is kept whole in its own chunk
    rather than cut."""
    target = chunk_chars if chunk_chars is not None else _chunk_chars_from_env()
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > target:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += added_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_chars_from_env() -> int:
    try:
        return int(os.environ.get("DISTIL_SUMMARY_CHUNK_CHARS", DEFAULT_CHUNK_CHARS))
    except ValueError:
        return DEFAULT_CHUNK_CHARS


def _max_retries_from_env() -> int:
    try:
        return int(os.environ.get("DISTIL_SUMMARY_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    except ValueError:
        return DEFAULT_MAX_RETRIES


def _min_chunk_summary_len(source_len: int) -> int:
    return max(MIN_CHUNK_SUMMARY_FLOOR, int(source_len * MIN_CHUNK_SUMMARY_RATIO))


def _min_merge_summary_len(source_len: int) -> int:
    return max(MIN_CHUNK_SUMMARY_FLOOR, int(source_len * MIN_MERGE_SUMMARY_RATIO))


def _complete_with_coverage_retry(
    prompt: str, system: str, minimum: int, client: LLMClient, max_retries: int, what: str
) -> str:
    """Shared retry loop for both the chunk and merge completions: a dropped connection and a
    too-thin result are both a failure worth retrying within the same bounded budget (mirrors
    ``extract.py``'s ``_complete_with_retry`` — a network exception and a response that fails
    validation are the same failure mode from the caller's perspective)."""
    last_detail = ""
    for attempt in range(max_retries):
        try:
            text = client.complete(prompt, system=system).strip()
        except Exception as exc:  # a dropped connection, retried like a thin result
            last_detail = f"completion failed ({exc})"
            logger.warning(
                "%s failed (attempt %d/%d): %s", what, attempt + 1, max_retries, last_detail
            )
            continue
        if len(text) >= minimum:
            return text
        last_detail = f"got {len(text)} chars, needed at least {minimum}"
        logger.warning(
            "%s too short (attempt %d/%d): %s", what, attempt + 1, max_retries, last_detail
        )
    raise NarrativeSummaryError(f"{what} failed after {max_retries} attempt(s): {last_detail}")


def _summarize_chunk(
    chunk: str, index: int, total: int, client: LLMClient, max_retries: int
) -> str:
    prompt = build_chunk_prompt(chunk, index, total)
    minimum = _min_chunk_summary_len(len(chunk))
    return _complete_with_coverage_retry(
        prompt, SYSTEM_CHUNK, minimum, client, max_retries, f"Chunk {index + 1}/{total} summary"
    )


def _merge_summaries(chunk_summaries: list[str], client: LLMClient, max_retries: int) -> str:
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    prompt = build_merge_prompt(chunk_summaries)
    source_len = sum(len(s) for s in chunk_summaries)
    minimum = _min_merge_summary_len(source_len)
    return _complete_with_coverage_retry(
        prompt, SYSTEM_MERGE, minimum, client, max_retries, "Merged summary"
    )


@dataclass
class NarrativeSummaryResult:
    text: str
    chunk_count: int
    model: str


def synthesize_narrative_summary(
    transcript_text: str,
    client: LLMClient,
    *,
    chunk_chars: int | None = None,
    max_retries: int | None = None,
    unslop_client: LLMClient | None = None,
) -> NarrativeSummaryResult:
    """Chunk ``transcript_text``, summarize each chunk, then merge into one narrative account.

    Raises :class:`NarrativeSummaryError` if the transcript has no usable content, or if a
    chunk or the merge stays below its coverage floor after the retry budget — this function
    never returns a silently-thin result. ``client`` is expected to be the cheap-tier client
    (see the module docstring); this function does not construct or select a model itself.
    """
    if not transcript_text.strip():
        raise NarrativeSummaryError("Transcript is empty; nothing to summarize.")

    chunks = chunk_transcript_text(transcript_text, chunk_chars)
    if not chunks:
        raise NarrativeSummaryError("Transcript produced no usable chunks.")

    retries = max_retries if max_retries is not None else _max_retries_from_env()
    summaries = [
        _summarize_chunk(chunk, i, len(chunks), client, retries)
        for i, chunk in enumerate(chunks)
    ]
    merged = _merge_summaries(summaries, client, retries)
    if unslop_client is not None:
        minimum = _min_merge_summary_len(len(merged))
        merged = rewrite_text(
            merged,
            unslop_client,
            validator=lambda text: len(text) >= minimum,
            max_attempts=retries,
        )
    model = getattr(client, "model", "") or resolve_stage_model("summary")
    return NarrativeSummaryResult(text=merged, chunk_count=len(chunks), model=model)
