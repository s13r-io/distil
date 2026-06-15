"""Stage 4 — Link knowledge to the profile. ARCHITECTURE.md §2; TESTING T-L1..L3.

The model proposes concrete application scenarios; the code enforces three invariants that do
not depend on the model behaving:

* **Goal validity (T-L1):** every link's ``linked_goal_id`` must be a real goal/focus id in
  the profile. Links naming an unknown id are dropped.
* **Novelty reservation (T-L2):** roughly ``DISTIL_NOVELTY_RATIO`` of links are flagged as
  novelty/serendipity (anti-filter-bubble), reserved deterministically if the model didn't.
* **Cold start (T-L3):** when ``meta.confidence`` is low, the prompt presents only the stable
  long-term goals (and active focus), never learned affinities — so a thin profile leans on
  what the user actually stated.
"""

from __future__ import annotations

from pydantic import ValidationError

from .extract import _parse_items_json  # reuse robust array parse
from .llm import LLMClient
from .models import ApplicationLink, KnowledgeItem, Profile
from .prompts.link import SYSTEM, build_link_prompt
from .triage import ParseError

_COLD_START_CONFIDENCE = 0.25


def valid_goal_ids(profile: Profile) -> set[str]:
    ids = {g.id for g in profile.stable.long_term_goals}
    ids |= {f.id for f in profile.current_focus}
    return ids


def generate_links(
    items: list[KnowledgeItem],
    profile: Profile,
    client: LLMClient,
    *,
    novelty_ratio: float = 0.2,
) -> list[ApplicationLink]:
    if not items:
        return []
    prompt = build_link_prompt(_render_goals(profile), _render_items(items))
    raw = client.complete(prompt, system=SYSTEM)
    proposed = _parse_links(raw)

    valid = valid_goal_ids(profile)
    kept = [link for link in proposed if link.linked_goal_id in valid]

    _reserve_novelty(kept, novelty_ratio)
    for i, link in enumerate(kept, start=1):
        link.link_id = f"a_{i:02d}"
    return kept


def _reserve_novelty(links: list[ApplicationLink], ratio: float) -> None:
    if ratio <= 0 or not links:
        # Honor an explicit zero ratio: clear any model-set novelty flags.
        if ratio <= 0:
            for link in links:
                link.novelty_flag = False
        return
    target = max(1, round(len(links) * ratio))
    already = sum(1 for link in links if link.novelty_flag)
    if already >= target:
        return
    # Flag additional links deterministically, spaced across the list.
    step = max(1, len(links) // target)
    flagged = already
    for idx in range(0, len(links), step):
        if flagged >= target:
            break
        if not links[idx].novelty_flag:
            links[idx].novelty_flag = True
            flagged += 1


def _parse_links(raw: str) -> list[ApplicationLink]:
    data = _parse_items_json(raw, kind="link")
    links: list[ApplicationLink] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ParseError("Each application link must be a JSON object.")
        obj.setdefault("link_id", f"a_{i + 1:02d}")
        try:
            links.append(ApplicationLink.model_validate(obj))
        except ValidationError as exc:
            raise ParseError(f"Application link {i} did not match the schema: {exc}") from exc
    return links


def _render_goals(profile: Profile) -> str:
    lines = []
    for g in profile.stable.long_term_goals:
        lines.append(f"- {g.id}: (long-term goal) {g.statement}")
    # Active focus is part of "what the user stated"; include it at cold start too.
    for f in profile.current_focus:
        if f.status == "active":
            lines.append(f"- {f.id}: (current focus) {f.project} — {f.description}")
    # Warm profiles may additionally hint at learned affinities; cold ones must not.
    if profile.meta.confidence >= _COLD_START_CONFIDENCE:
        top = sorted(profile.affinities.topics.items(), key=lambda kv: -kv[1])[:5]
        if top:
            hint = ", ".join(f"{k} ({v:.1f})" for k, v in top)
            lines.append(f"(learned topic affinities, for flavor only: {hint})")
    return "\n".join(lines) if lines else "(no goals on file)"


def _render_items(items: list[KnowledgeItem]) -> str:
    return "\n".join(f"- {it.item_id} [{it.type}]: {it.statement}" for it in items)
