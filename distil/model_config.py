"""Per-stage model resolution — the general mechanism for choosing which model string each
LLM-calling pipeline stage uses, so that changing one stage's model later is a matter of
setting an environment variable, never a code change.

Every stage that makes a model call resolves its own model independently, by name:
``DISTIL_MODEL_<STAGE>`` (the stage name upper-cased) wins if set; otherwise the stage falls
back to its tier default. Every stage in :data:`STRONG_TIER_STAGES` falls back to
``DISTIL_MODEL`` — the existing, single strong-tier setting — while every stage in
:data:`CHEAP_TIER_STAGES` falls back to a hardcoded cheap-tier default instead. The credential is
unaffected either way: every stage still reads ``ANTHROPIC_API_KEY`` the same way
(:class:`distil.llm.AnthropicClient` does that lazily, regardless of which model string it was
constructed with).

**Tier assignment is measured, not assumed (owner decision).** On a real ~11,719-word
transcript: model tier decided faithfulness at extraction and only at extraction — all-Sonnet
kept 19/19 extracted items as faithful, all-Haiku kept only 8/16 (half dropped as unfaithful
quotes) — so **extraction** stays strong-tier; that same measurement showed a mixed
strong-extract/cheap-elsewhere run kept faithfulness (19/21) at roughly a third of the
all-Sonnet cost. **Triage** is a coarse categorical judgement (dominant type / density /
transcript-loss), not the verbatim-quote-faithfulness work — cheap-tier. **Link** and **note**
synthesize from already-extracted, already-faithful items rather than re-reading the raw
transcript for quotable provenance — cheap-tier, same reasoning as ``summary`` (see
``distil/summary.py``'s module docstring). **Canonicalize** stays strong-tier: it makes the
match/new/reject judgement call for each item against the existing knowledge base, a decision
closer in kind to extraction's than to link/note's templated synthesis. ``graph`` was not
covered by this measurement and is left on its prior default (strong) rather than downgraded
without evidence.

This module replaces the one-off ``DISTIL_SUMMARY_MODEL`` global with the same
``DISTIL_MODEL_<STAGE>`` convention every other stage now uses — a second ad hoc global
variable per new cheap-tier stage would not have generalized.

Scope note: this is the resolution mechanism plus a client factory (:func:`make_stage_client`)
built on it. ``cli.py`` and ``web/app.py`` each construct every stage's client via its own
``_make_<stage>_client()`` seam on top of :func:`make_stage_client` and pass it into
``pipeline.run_pipeline``'s matching per-stage keyword argument — see ``run_pipeline``'s
docstring. A settings UI for editing these values is a deliberate follow-up; the wiring itself
is not.
"""

from __future__ import annotations

import os

from .llm import AnthropicClient, LLMClient

# Current Haiku-tier model (resolved from Anthropic's own catalogue, not guessed — see the PR
# that introduced the narrative summary layer for the lookup). Kept as the name every cheap-tier
# stage's default resolves to, not just summary's — see DEFAULT_CHEAP_TIER_MODEL below.
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5"

# Same value, generalized name — every CHEAP_TIER_STAGES member defaults to this, not just
# "summary" (kept as the one constant so a future retune of the cheap tier is one edit).
DEFAULT_CHEAP_TIER_MODEL = DEFAULT_SUMMARY_MODEL

# Stages that share today's single strong-tier default (DISTIL_MODEL) unless individually
# overridden via DISTIL_MODEL_<STAGE>. Kept as an explicit tuple — not "anything not cheap" —
# so adding a stage here is a deliberate decision, not an accident of exclusion. See the module
# docstring's "Tier assignment is measured, not assumed" section for why each stage landed here.
STRONG_TIER_STAGES = ("extract", "canonicalize", "graph")

# Stages that default to a cheaper tier unless individually overridden. See the module
# docstring for the measurement behind each of these.
CHEAP_TIER_STAGES = ("summary", "triage", "link", "note")

_CHEAP_TIER_DEFAULTS = {stage: DEFAULT_CHEAP_TIER_MODEL for stage in CHEAP_TIER_STAGES}


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
# hardcoded max_tokens=4096 for every stage; on a real ~1,455-word transcript the (then-merged)
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
# extracted knowledge item (per chunk, when extraction is chunked — see extract.py's module
# docstring) — its ceiling instead resolves to the model's own real maximum, so a dense chunk's
# full item list is never silently cut off, costing nothing on a typical chunk that needs far less.
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
