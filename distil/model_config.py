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
closer in kind to extraction's than to link/note's templated synthesis. **``graph`` moved to the
cheap tier (owner decision, corrects the record rather than a re-measurement).** It was
originally placed on the strong tier by analogy with canonicalize, not by measurement — it never
actually ran in the three-way tier comparison above, because that test store had no candidate
entries and ``link_graph`` returned early before making any model call. Its real call shape (see
``distil/graph.py``) is a small pairwise relation classification — one call per related
candidate, classifying a single label out of a fixed enum (``contrasts_with``/``builds_on``/
``related``/``none``) from two short summaries — which is cheap-tier work like link/note, not
extraction-grade verbatim-quote faithfulness, and its cost scales with library size (one call per
candidate) rather than per-video, so keeping it cheap matters more as the library grows.

This module replaces the one-off ``DISTIL_SUMMARY_MODEL`` global with the same
``DISTIL_MODEL_<STAGE>`` convention every other stage now uses — a second ad hoc global
variable per new cheap-tier stage would not have generalized.

Scope note: this is the resolution mechanism plus a client factory (:func:`make_stage_client`)
built on it. ``cli.py`` and ``web/app.py`` each construct every stage's client via its own
``_make_<stage>_client()`` seam on top of :func:`make_stage_client` and pass it into
``pipeline.run_pipeline``'s matching per-stage keyword argument — see ``run_pipeline``'s
docstring.

**Settings-UI precedence (Phase F).** ``resolve_stage_model`` checks, in order: a stored setting
(:class:`distil.model_settings.ModelSettingsStore`, persisted beside the database on the mounted
volume — the web settings page at ``/settings`` writes here) > ``DISTIL_MODEL_<STAGE>`` > the
tier default described above. The stored-setting lookup is looked up fresh on every call (a cheap
sqlite read, and a no-op when the db file doesn't exist yet) — nothing is cached at import time —
so a change made from the settings page takes effect on the very next ``make_stage_client(stage)``
call, i.e. the next video; a pipeline run already holding an already-constructed ``LLMClient``
keeps whatever model that client was built with, since :class:`~distil.llm.AnthropicClient` fixes
its model string at construction (see ``distil/llm.py``) and never re-resolves it mid-call.
:func:`resolve_stage_model_info` reports which of the three sources is currently in force for a
stage, for the settings page to display; :mod:`distil.model_settings` owns storage and model-string
validation, imported here lazily (inside the functions that need it) to avoid a module import cycle
(``model_settings`` imports :data:`MODEL_MAX_OUTPUT_TOKENS` from this module at import time).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
STRONG_TIER_STAGES = ("extract", "canonicalize")

# Stages that default to a cheaper tier unless individually overridden. See the module
# docstring for the measurement behind each of these.
CHEAP_TIER_STAGES = ("summary", "triage", "link", "note", "graph")

# Every model-calling stage, in the order the settings surface presents them — extract first
# since it's the one stage whose model choice has a measured, stated faithfulness cost (see
# EXTRACT_CHEAP_MODEL_WARNING below), not derived from the tier tuples' own order.
ALL_STAGES = ("extract", "canonicalize", "graph", "triage", "link", "note", "summary")

_CHEAP_TIER_DEFAULTS = {stage: DEFAULT_CHEAP_TIER_MODEL for stage in CHEAP_TIER_STAGES}

# Sources a stage's resolved model can come from, in precedence order — see the module
# docstring's "Settings-UI precedence" section.
SOURCE_STORED = "stored"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"
SOURCE_UNCONFIGURED = "unconfigured"

# Measured cost of running extraction on a cheap-tier model (see the module docstring's "Tier
# assignment is measured, not assumed" section) — surfaced by the settings page whenever the
# owner points "extract" at a Haiku-family model, without blocking the choice.
EXTRACT_CHEAP_MODEL_WARNING = (
    "Extraction on a cheap-tier (Haiku) model measured 8 of 16 items kept as faithful quotes, "
    "versus 19 of 19 on the strong tier — roughly half dropped as unfaithful. This will still "
    "be applied; it is not blocked."
)


def resolve_stage_model(stage: str) -> str:
    """The model string ``stage`` should use, checked in precedence order: a stored setting
    (:class:`distil.model_settings.ModelSettingsStore`) > ``DISTIL_MODEL_<STAGE>`` > the stage's
    tier default (``DISTIL_MODEL`` for a strong-tier stage, a hardcoded cheap-tier default for a
    cheap-tier one). Raises ``RuntimeError`` for a strong-tier stage with no stored setting, no
    per-stage override, and no ``DISTIL_MODEL`` configured — the same "must be set explicitly"
    contract :class:`AnthropicClient` already enforces, just resolved per stage instead of
    globally."""
    from .model_settings import ModelSettingsStore

    stored = ModelSettingsStore().get(stage)
    if stored:
        return stored
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


@dataclass(frozen=True)
class StageModelInfo:
    """What the settings page needs to render one stage's row: the model currently in force,
    which of the three precedence sources produced it, and the env var name that would override
    it. ``model`` is ``None`` only for :data:`SOURCE_UNCONFIGURED` (a strong-tier stage with
    nothing set anywhere). ``active_env_var`` is set only when ``source == SOURCE_ENV`` — it
    names the *specific* env var supplying the value, which for a strong-tier stage falling
    back to the global default is ``DISTIL_MODEL``, not ``env_var`` (the per-stage override
    name, always reported so the UI can show "override with ..." regardless of whether it's
    the active source)."""

    stage: str
    model: str | None
    source: str
    env_var: str
    active_env_var: str | None
    faithfulness_warning: str | None


def resolve_stage_model_info(stage: str) -> StageModelInfo:
    """Same precedence :func:`resolve_stage_model` applies, but reports the source instead of
    raising — a display-only read for the settings page, never used by the pipeline itself."""
    from .model_settings import ModelSettingsStore

    env_var = f"DISTIL_MODEL_{stage.upper()}"
    active_env_var: str | None = None
    stored = ModelSettingsStore().get(stage)
    if stored:
        model, source = stored, SOURCE_STORED
    else:
        override = os.environ.get(env_var)
        if override:
            model, source, active_env_var = override, SOURCE_ENV, env_var
        elif stage in CHEAP_TIER_STAGES:
            model, source = _CHEAP_TIER_DEFAULTS[stage], SOURCE_DEFAULT
        else:
            global_default = os.environ.get("DISTIL_MODEL", "")
            if global_default:
                model, source, active_env_var = global_default, SOURCE_ENV, "DISTIL_MODEL"
            else:
                model, source = None, SOURCE_UNCONFIGURED
    warning = (
        EXTRACT_CHEAP_MODEL_WARNING
        if (stage == "extract" and model is not None and "haiku" in model.lower())
        else None
    )
    return StageModelInfo(
        stage=stage,
        model=model,
        source=source,
        env_var=env_var,
        active_env_var=active_env_var,
        faithfulness_warning=warning,
    )


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
