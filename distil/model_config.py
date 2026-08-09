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

Scope note: this is the resolution *mechanism* plus a client factory built on it. Actually
wiring distinct per-stage clients into ``pipeline.run_pipeline`` for the six existing strong-
tier stages (today they still share one injected client, unchanged), and any settings UI for
editing these values, are deliberate follow-ups — see ``run_pipeline``'s docstring for the
optional per-stage keyword arguments this mechanism is meant to feed.
"""

from __future__ import annotations

import os

from .llm import AnthropicClient, LLMClient

# Current Haiku-tier model (resolved from Anthropic's own catalogue, not guessed — see the PR
# that introduced the narrative summary layer for the lookup).
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5"

# Stages that share today's single strong-tier default (DISTIL_MODEL) unless individually
# overridden via DISTIL_MODEL_<STAGE>. Kept as an explicit tuple — not "anything not cheap" —
# so adding a stage here is a deliberate decision, not an accident of exclusion.
STRONG_TIER_STAGES = ("triage", "extract", "link", "note", "graph", "canonicalize")

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


def make_stage_client(stage: str, *, api_key: str | None = None) -> LLMClient:
    """Construct the :class:`LLMClient` for ``stage``, with its model resolved via
    :func:`resolve_stage_model`. The credential is the same every time —
    :class:`AnthropicClient` reads ``ANTHROPIC_API_KEY`` itself (lazily, at the first real
    call) whenever ``api_key`` is omitted, exactly as it already does today."""
    return AnthropicClient(model=resolve_stage_model(stage), api_key=api_key)
