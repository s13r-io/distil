"""Non-blocking two-pass style rewriting for already-grounded prose.

The guide is package data and is read on every rewrite so editing its markdown changes behavior
without a code change. Callers must inject the ``summary``-tier client; a concrete client whose
public model differs from ``resolve_stage_model("summary")`` is rejected rather than silently
spending a strong-tier call.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .model_config import resolve_stage_model

logger = logging.getLogger(__name__)

STYLE_GUIDE_PATH = Path(__file__).parent / "prompts" / "style_guides" / "unslop.md"
_FALSE_VALUES = {"0", "false", "no", "off"}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def unslop_enabled() -> bool:
    """Return the env-controlled switch; unset and unrecognized values keep it enabled."""
    return os.environ.get("DISTIL_UNSLOP_ENABLED", "true").strip().lower() not in _FALSE_VALUES


def _guide() -> str:
    return STYLE_GUIDE_PATH.read_text(encoding="utf-8")


def _uses_summary_tier(client: LLMClient) -> bool:
    model = getattr(client, "model", "")
    return not model or model == resolve_stage_model("summary")


def _rewrite_prompt(text: str, guide: str) -> str:
    return f"""Rewrite the input text following the attached style guide.

Preserve meaning and facts exactly. Never invent a claim, example, source, number, or detail
that is not present in the input. Return only the rewritten text, with no commentary.

<style-guide>
{guide}
</style-guide>

<input>
{text}
</input>"""


def _audit_prompt(text: str, guide: str) -> str:
    return f"""Self-audit the draft using the attached style guide. Ask: "What in this still reads as obviously AI-written?" Fix remaining tells and return only the final text.

Preserve meaning and facts exactly. Never invent a claim, example, source, number, or detail
that is not present in the draft. Return no audit notes or commentary.

<style-guide>
{guide}
</style-guide>

<draft>
{text}
</draft>"""


def rewrite_text(
    text: str,
    client: LLMClient,
    *,
    validator: Callable[[str], bool] | None = None,
    max_attempts: int = 1,
) -> str:
    """Run the guide's rewrite and self-audit calls, falling back to ``text`` on any failure.

    ``max_attempts`` retries the complete two-call pass when a caller-specific validator rejects
    the result. This is used by narrative summary's existing coverage floor.
    """
    if not text.strip() or not unslop_enabled():
        return text
    try:
        if not _uses_summary_tier(client):
            logger.warning("Unslop skipped because its client is not on the summary model tier.")
            return text
        guide = _guide()
    except Exception:
        logger.exception("Unslop setup failed; keeping the original prose.")
        return text

    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            first = client.complete(_rewrite_prompt(text, guide)).strip()
            if not first:
                raise ValueError("rewrite pass returned empty text")
            final = client.complete(_audit_prompt(first, guide)).strip()
            if not final:
                raise ValueError("self-audit pass returned empty text")
            if validator is None or validator(final):
                return final
            logger.warning(
                "Unslop validation rejected output (attempt %d/%d).", attempt + 1, attempts
            )
        except Exception:
            logger.warning(
                "Unslop failed (attempt %d/%d); keeping the original prose.",
                attempt + 1,
                attempts,
                exc_info=True,
            )
            return text
    return text


def rewrite_json_fields(
    value: Any,
    client: LLMClient,
    *,
    text_keys: set[str],
    id_keys: set[str],
) -> Any:
    """Batch-rewrite selected JSON string fields while preserving structure and IDs exactly."""
    original_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def validate(candidate_text: str) -> bool:
        try:
            candidate = _parse_json(candidate_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return _id_arrays(value, id_keys) == _id_arrays(candidate, id_keys) and _only_text_changed(
            value, candidate, text_keys
        )

    rewritten = rewrite_text(original_text, client, validator=validate)
    if rewritten == original_text:
        return value
    try:
        candidate = _parse_json(rewritten)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
    return candidate if validate(rewritten) else value


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return json.loads(stripped)


def _id_arrays(value: Any, id_keys: set[str], path: tuple[Any, ...] = ()) -> list[tuple]:
    found: list[tuple] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, key)
            if key in id_keys:
                found.append((child_path, json.dumps(child, separators=(",", ":"))))
            found.extend(_id_arrays(child, id_keys, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_id_arrays(child, id_keys, (*path, index)))
    return found


def _only_text_changed(original: Any, candidate: Any, text_keys: set[str]) -> bool:
    if type(original) is not type(candidate):
        return False
    if isinstance(original, dict):
        if original.keys() != candidate.keys():
            return False
        for key in original:
            if key in text_keys:
                if not isinstance(original[key], str) or not isinstance(candidate[key], str):
                    return False
            elif not _only_text_changed(original[key], candidate[key], text_keys):
                return False
        return True
    if isinstance(original, list):
        return len(original) == len(candidate) and all(
            _only_text_changed(left, right, text_keys)
            for left, right in zip(original, candidate, strict=True)
        )
    return original == candidate
