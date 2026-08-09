"""Per-stage model resolution — the general mechanism for choosing which model string each
LLM-calling pipeline stage uses, so that changing one stage's model later is a matter of
setting an environment variable, never a code change.

Every stage that makes a model call resolves its own model independently, by name:
``DISTIL_MODEL_<STAGE>`` (the stage name upper-cased) wins if set; otherwise the stage falls
back to its tier default. Today every stage in :data:`STRONG_TIER_STAGES` falls back to
``DISTIL_MODEL`` — the existing, single strong-tier setting, so this preserves current
behavior byte-for-byte for every stage that isn't individually overridden. Only
:data:`CHEAP_TIER_STAGES` (currently just ``"summary"``) falls back to a cheap tier default
instead — see ``distil/summary.py``'s module docstring for why that layer is deliberately
cheaper. The credential is unaffected either way: every stage still reads
``ANTHROPIC_API_KEY`` the same way (:class:`distil.llm.AnthropicClient` does that lazily,
regardless of which model string it was constructed with).

This module replaces the one-off ``DISTIL_SUMMARY_MODEL`` global with the same
``DISTIL_MODEL_<STAGE>`` convention every other stage now uses — a second ad hoc global
variable per new cheap-tier stage would not have generalized.

Scope note: this is the resolution mechanism plus a client factory (:func:`make_stage_client`)
built on it. ``cli.py`` and ``web/app.py`` each construct every strong-tier stage's client via
its own ``_make_<stage>_client()`` seam on top of :func:`make_stage_client` and pass it into
``pipeline.run_pipeline``'s matching per-stage keyword argument — see ``run_pipeline``'s
docstring. A settings UI for editing these values is a deliberate follow-up; the wiring itself
is not.
"""

from __future__ import annotations

import os

from .llm import AnthropicClient, LLMClient

# Current Haiku-tier model (resolved from Anthropic's own catalogue, not guessed — see the PR
# that introduced the narrative summary layer for the lookup).
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5"

# Stages that share today's single strong-tier default (DISTIL_MODEL) unless individually
# overridden via DISTIL_MODEL_<STAGE>. Kept as an explicit tuple — not "anything not cheap" —
# so adding a stage here is a deliberate decision, not an accident of exclusion. "triage" is not
# a stage of its own anymore: triage and extraction are merged into one call, resolved as
# "extract" (see pipeline.py's and extract.py's module docstrings).
STRONG_TIER_STAGES = ("extract", "link", "note", "graph", "canonicalize")

# Stages that default to a cheaper tier unless individually overridden.
CHEAP_TIER_STAGES = ("summary",)

_CHEAP_TIER_DEFAULTS = {"summary": DEFAULT_SUMMARY_MODEL}


def resolve_stage_model(stage: str) -> str:
    """The model string ``stage`` should use: ``DISTIL_MODEL_<STAGE>`` if set, else the
    stage's tier default (``DISTIL_MODEL`` for a strong-tier stage, a hardcoded cheap-tier
    default for a cheap-tier one). Raises ``RuntimeError`` for a strong-tier stage with no
    ``DISTIL_MODEL`` configured — the same "must be set explicitly" contract
    :class:`AnthropicClient` already enforces, just resolved per stage instead of globally."""
    override = os.environ.get(f"DISTIL_MODEL_{stage.upper()}")
    if override:
        return override
    if stage in CHEAP_TIER_STAGES:
        return _CHEAP_TIER_DEFAULTS[stage]
    value = os.environ.get("DISTIL_MODEL", "")
    if not value:
        raise RuntimeError(
            "DISTIL_MODEL is not set. Set it in your .env to a current model string "
            "(see .env.example) — Distil does not hardcode a model."
        )
    return value


# Per-stage max_tokens ceiling — the same "resolve per stage by name" mechanism as
# resolve_stage_model, not a second global constant. AnthropicClient.complete() previously
# hardcoded max_tokens=4096 for every stage; on a real ~1,455-word transcript the merged
# triage+extract call hit that ceiling *exactly* and silently lost items to the salvage path
# (see the PR that introduced this mechanism for the measurement). max_tokens is an output
# *ceiling*, not a spend — the API only bills for tokens actually generated, so raising it here
# costs nothing unless a response actually needs the extra room.
#
# Published per-model output ceilings ("max_tokens" in the Models API / platform.claude.com
# model catalog — https://platform.claude.com/docs/en/about-claude/models/overview, and
# confirmed via the Models API's ``max_tokens`` field), not guessed:
#   claude-sonnet-5:   128,000 max output tokens
#   claude-haiku-4-5:   64,000 max output tokens
# These are the two models Distil actually configures today (DISTIL_MODEL=claude-sonnet-5 for
# the strong tier, DEFAULT_SUMMARY_MODEL=claude-haiku-4-5 for the cheap tier).
MODEL_MAX_OUTPUT_TOKENS = {
    "claude-sonnet-5": 128_000,
    "claude-haiku-4-5": 64_000,
}

# Every stage except "extract" keeps the pre-existing flat 4096 ceiling — a triage classification,
# a handful of application links, or a short synthesized note doesn't need headroom that scales
# with transcript length. "extract" is the one stage whose output is genuinely one JSON object per
# extracted knowledge item across a merged triage+extraction call spanning the *whole* transcript
# (see extract.py's module docstring) — its ceiling instead resolves to the model's own real
# maximum, so a long video's full item list is never silently cut off again.
_DEFAULT_STAGE_MAX_TOKENS = 4096
_UNCAPPED_STAGES = ("extract",)


def resolve_stage_max_tokens(stage: str) -> int:
    """The max_tokens ceiling ``stage`` should request: ``DISTIL_MAX_TOKENS_<STAGE>`` if set,
    else the stage's own default (the resolved model's published output ceiling for
    ``_UNCAPPED_STAGES``, or ``_DEFAULT_STAGE_MAX_TOKENS`` for every other stage — unchanged
    from the previous flat hardcoded value). An unrecognized model string for an uncapped stage
    falls back to ``_DEFAULT_STAGE_MAX_TOKENS`` rather than guessing a larger number."""
    override = os.environ.get(f"DISTIL_MAX_TOKENS_{stage.upper()}")
    if override:
        return int(override)
    if stage not in _UNCAPPED_STAGES:
        return _DEFAULT_STAGE_MAX_TOKENS
    model = resolve_stage_model(stage)
    return MODEL_MAX_OUTPUT_TOKENS.get(model, _DEFAULT_STAGE_MAX_TOKENS)


def make_stage_client(stage: str, *, api_key: str | None = None) -> LLMClient:
    """Construct the :class:`LLMClient` for ``stage``, with its model resolved via
    :func:`resolve_stage_model` and its output ceiling resolved via
    :func:`resolve_stage_max_tokens`. The credential is the same every time —
    :class:`AnthropicClient` reads ``ANTHROPIC_API_KEY`` itself (lazily, at the first real
    call) whenever ``api_key`` is omitted, exactly as it already does today."""
    return AnthropicClient(
        model=resolve_stage_model(stage),
        api_key=api_key,
        max_tokens=resolve_stage_max_tokens(stage),
    )
