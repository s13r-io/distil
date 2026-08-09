"""Narrative summary prompts (summary layer). See ``distil/summary.py``'s module docstring for
why this exists alongside ``prompts/note.py`` rather than instead of it: this reads the whole
transcript chunk by chunk, with nothing filtered first, to recover the build-up, examples, and
digressions that atomic extraction drops.
"""

from __future__ import annotations

PROMPT_VERSION = "summary/v1"

SYSTEM_CHUNK = (
    "You write a narrative, flowing summary of one chunk of a spoken transcript. Cover the "
    "argument's build-up, the reasoning, worked examples, and any digressions that explain the "
    "idea — do not just list facts or reduce the passage to a single takeaway. Write plain "
    "prose paragraphs, no headings, no bullet lists, no markdown. Respond with the summary text "
    "only, nothing else."
)

_CHUNK_TEMPLATE = """\
This is chunk {index} of {total} from one video's transcript, in order.

TRANSCRIPT CHUNK:
{chunk}

Write a thorough narrative summary of this chunk. It should read as a flowing account a \
person could follow on its own, covering the substance in the order it was discussed — the \
setup, the reasoning, any examples, and digressions that add real explanation. Do not \
compress this into one or two sentences; a chunk this long deserves a proportionally \
substantial summary."""


def build_chunk_prompt(chunk: str, index: int, total: int) -> str:
    return _CHUNK_TEMPLATE.format(chunk=chunk, index=index + 1, total=total)


SYSTEM_MERGE = (
    "You merge several sequential chunk summaries of one video's transcript into a single, "
    "coherent, flowing narrative account. Preserve chronological order and the connective "
    "tissue between sections — smooth over chunk boundaries rather than concatenating them, "
    "and drop only redundancy introduced by the chunking itself, not substance. Write plain "
    "prose paragraphs, no headings, no bullet lists, no markdown. Respond with the merged "
    "narrative text only, nothing else."
)

_MERGE_TEMPLATE = """\
CHUNK SUMMARIES, IN ORDER:
{summaries}

Weave these into one coherent narrative account of the whole video, in the order given. Keep \
the substance of each chunk — this is a merge into one flowing read, not a further \
compression down to a few sentences."""


def build_merge_prompt(chunk_summaries: list[str]) -> str:
    numbered = "\n\n".join(
        f"[{i + 1}] {summary}" for i, summary in enumerate(chunk_summaries)
    )
    return _MERGE_TEMPLATE.format(summaries=numbered)
