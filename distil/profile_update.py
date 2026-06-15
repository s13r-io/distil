"""Stage 7 — Feedback → Profile update (PURE). SCHEMA.md §3; TESTING T-P1..P8.

The crux of personalization: the *same* score teaches opposite lessons depending on the
``reason``. Updates are gradual (EMA toward a target, rate ``alpha``) and bounded — a single
event can never swing a weight past 1.0. The function is pure: it returns a new
:class:`Profile` and never mutates its input.

| score | reason         | update                                                            |
|-------|----------------|-------------------------------------------------------------------|
| 4–5   | relevant       | upweight tags (topics/types/forms) tied to linked goal            |
| 4–5   | (novelty link) | add a new affinity — discovered an undeclared interest            |
| 1–2   | bad_source     | nothing about the user (source-quality model only)                |
| 1–2   | already_knew   | add topic to known_topics; not a negative                         |
| 1–2   | wrong_for_me   | upweight matching negatives dimension                             |
| 1–2   | irrelevant_now | soft current-focus signal only                                    |
| 3     | any            | small/zero delta                                                  |
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import KBEntry, NegativeEntry, Profile

# A lukewarm (score 3) event applies only a small fraction of the normal step.
_LUKEWARM_DAMPING = 0.1
_CONFIDENCE_CEILING = 1.0
_CONFIDENCE_DOCS_FULL = 30  # confidence approaches 1.0 around this many documents


def _ema(current: float, target: float, alpha: float) -> float:
    """Exponential moving average step toward ``target``; result stays in [0, target]."""
    return current + alpha * (target - current)


def apply_feedback(profile: Profile, entry: KBEntry, *, alpha: float = 0.3) -> Profile:
    out = profile.model_copy(deep=True)
    fb = entry.feedback

    # Always advance the document counter + confidence (even for bad_source: we did process one).
    out.meta.documents_processed = profile.meta.documents_processed + 1
    # Confidence rises monotonically with documents processed (never drops).
    out.meta.confidence = max(
        profile.meta.confidence,
        min(_CONFIDENCE_CEILING, out.meta.documents_processed / _CONFIDENCE_DOCS_FULL),
    )
    out.meta.last_updated = datetime.now(timezone.utc).isoformat()

    score = fb.score
    reason = fb.reason
    if score is None:
        return out  # nothing scored yet

    topics = entry.tags.topics
    types = entry.tags.knowledge_types
    forms = entry.tags.application_forms

    # Lukewarm (3): tiny nudge in the "relevant" direction, then stop (T-P7).
    if score == 3:
        step = alpha * _LUKEWARM_DAMPING
        _upweight_affinities(out, topics, types, forms, alpha=step)
        return out

    positive = score >= 4
    negative = score <= 2

    if positive:
        # 4–5 relevant: upweight the tag dimensions tied to the linked goal (T-P1).
        _upweight_affinities(out, topics, types, forms, alpha=alpha)
        # 4–5 with a novelty link that paid off: the novel topics become new affinities (T-P6).
        _absorb_novelty(out, entry, alpha=alpha)
        return out

    if negative:
        if reason == "bad_source":
            # Update nothing about the user (T-P2). Source-quality model is out of scope for
            # the profile object; a real impl would log it separately.
            return out
        if reason == "already_knew":
            # Mark known; suppress basics later. Not a negative (T-P3).
            for topic in topics:
                if topic not in out.known_topics:
                    out.known_topics.append(topic)
            return out
        if reason == "wrong_for_me":
            # Genuine mismatch → upweight negatives, reason-tagged (T-P4).
            _upweight_negatives(out, topics, types, forms, alpha=alpha, reason="wrong_for_me")
            return out
        if reason == "irrelevant_now":
            # Soft, timing-related: touch current_focus recency only; no topic dislike (T-P5).
            _soft_focus_touch(out)
            return out

    return out


def _upweight_affinities(profile, topics, types, forms, *, alpha) -> None:
    for topic in topics:
        cur = profile.affinities.topics.get(topic, 0.0)
        profile.affinities.topics[topic] = round(_ema(cur, 1.0, alpha), 6)
    for t in types:
        cur = profile.affinities.knowledge_types.get(t, 0.0)
        profile.affinities.knowledge_types[t] = round(_ema(cur, 1.0, alpha), 6)
    for f in forms:
        cur = profile.affinities.application_forms.get(f, 0.0)
        profile.affinities.application_forms[f] = round(_ema(cur, 1.0, alpha), 6)


def _absorb_novelty(profile, entry, *, alpha) -> None:
    """A positively rated novelty link reveals an undeclared interest → seed a new affinity.

    ``_upweight_affinities`` already nudged every tag dimension. This step additionally seeds
    affinities for the *forms* a novelty link introduced (e.g. an "experiment" the user had
    never engaged with), so serendipity that pays off broadens the profile rather than just
    reinforcing what was already declared (SCHEMA §3, row "novelty link").
    """
    novelty_forms = {
        link.application_form for link in entry.application_links if link.novelty_flag
    }
    for form in novelty_forms:
        if form not in profile.affinities.application_forms:
            profile.affinities.application_forms[form] = round(_ema(0.0, 1.0, alpha), 6)


def _upweight_negatives(profile, topics, types, forms, *, alpha, reason) -> None:
    def bump(bucket: dict[str, NegativeEntry], keys) -> None:
        for k in keys:
            entry = bucket.get(k) or NegativeEntry()
            entry.weight = round(min(1.0, _ema(entry.weight, 1.0, alpha)), 6)
            entry.reasons[reason] = entry.reasons.get(reason, 0) + 1
            bucket[k] = entry

    bump(profile.negatives.topics, topics)
    bump(profile.negatives.knowledge_types, types)
    bump(profile.negatives.application_forms, forms)


def _soft_focus_touch(profile) -> None:
    """irrelevant_now is a timing signal: nudge focus recency, do not learn a topic dislike."""
    now = datetime.now(timezone.utc).isoformat()
    for focus in profile.current_focus:
        if focus.status == "active":
            focus.last_touched = now
