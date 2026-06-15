"""Phase 2.1 — profile_update.py (PURE). One test per SCHEMA §3 row. Tests T-P1..P8.

The same score teaches opposite lessons depending on the reason. These tests pin each row of
the SCHEMA §3 table. Updates are EMA-bounded (no single event swings a weight past a cap).
"""

import copy

import pytest

from distil.models import KBEntry, Profile
from distil.profile_update import apply_feedback


def _profile(**over) -> Profile:
    base = {
        "user_id": "owner",
        "stable": {
            "role": "engineer",
            "domain": "backend",
            "long_term_goals": [
                {"id": "g_01", "statement": "ship reliable services", "created_at": "2026-01-01T00:00:00"}
            ],
        },
        "meta": {"documents_processed": 5, "confidence": 0.3},
    }
    base.update(over)
    return Profile.model_validate(base)


def _entry(score, reason, *, topics=("kubernetes",), types=("heuristic",), forms=("trigger",),
           linked_goal="g_01", novelty_links=None, per_link=None) -> KBEntry:
    links = []
    for i, form in enumerate(forms):
        links.append({
            "link_id": f"a_{i}",
            "knowledge_item_ids": ["k_01"],
            "linked_goal_id": linked_goal,
            "application_form": form,
            "scenario": "do the thing",
            "novelty_flag": False,
        })
    for j, (form, _topic) in enumerate(novelty_links or []):
        links.append({
            "link_id": f"n_{j}",
            "knowledge_item_ids": ["k_01"],
            "linked_goal_id": linked_goal,
            "application_form": form,
            "scenario": "serendipity",
            "novelty_flag": True,
        })
    fb = {"score": score, "reason": reason}
    if per_link:
        fb["per_link"] = per_link
    return KBEntry.model_validate({
        "entry_id": "e_01",
        "source": {"title": "t", "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [{
            "item_id": "k_01", "type": "heuristic", "statement": "s", "stance": "opinion",
            "provenance": {"quote": "q"},
        }],
        "application_links": links,
        "tags": {"topics": list(topics) + [t for _, t in (novelty_links or [])],
                 "knowledge_types": list(types),
                 "application_forms": list(forms) + [f for f, _ in (novelty_links or [])]},
        "feedback": fb,
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    })


# ---- T-P1: (5, relevant) upweights topics/types/forms tied to the linked goal ----


@pytest.mark.unit
def test_p1_relevant_upweights_tag_dimensions():
    p = _profile()
    out = apply_feedback(p, _entry(5, "relevant"), alpha=0.3)
    assert out.affinities.topics.get("kubernetes", 0) > 0
    assert out.affinities.knowledge_types.get("heuristic", 0) > 0
    assert out.affinities.application_forms.get("trigger", 0) > 0
    assert out.meta.documents_processed == p.meta.documents_processed + 1


# ---- T-P2: (1, bad_source) leaves the user profile byte-identical (only source model) ----


@pytest.mark.unit
def test_p2_bad_source_does_not_touch_user_model():
    p = _profile()
    before = copy.deepcopy(p)
    out = apply_feedback(p, _entry(1, "bad_source"), alpha=0.3)
    # user-learned dimensions unchanged
    assert out.affinities == before.affinities
    assert out.negatives == before.negatives
    assert out.known_topics == before.known_topics


# ---- T-P3: (2, already_knew) adds topic to known_topics; NOT to negatives ----


@pytest.mark.unit
def test_p3_already_knew_marks_known_not_negative():
    p = _profile()
    out = apply_feedback(p, _entry(2, "already_knew"), alpha=0.3)
    assert "kubernetes" in out.known_topics
    assert "kubernetes" not in out.negatives.topics


# ---- T-P4: (1, wrong_for_me) increments the matching negatives dimension ----


@pytest.mark.unit
def test_p4_wrong_for_me_upweights_negatives():
    p = _profile()
    out = apply_feedback(p, _entry(1, "wrong_for_me"), alpha=0.3)
    neg = out.negatives.topics.get("kubernetes")
    assert neg is not None and neg.weight > 0
    assert neg.reasons.get("wrong_for_me", 0) == 1


# ---- T-P5: (1, irrelevant_now) applies only a soft/current-focus adjustment ----


@pytest.mark.unit
def test_p5_irrelevant_now_is_soft_no_negative_no_known():
    p = _profile()
    out = apply_feedback(p, _entry(1, "irrelevant_now"), alpha=0.3)
    assert "kubernetes" not in out.negatives.topics
    assert "kubernetes" not in out.known_topics
    # topic affinity not driven negative or strongly positive
    assert out.affinities.topics.get("kubernetes", 0.0) == 0.0


# ---- T-P6: (5, novelty link) adds a new affinity not previously present ----


@pytest.mark.unit
def test_p6_novelty_link_adds_new_affinity():
    p = _profile()
    entry = _entry(5, "relevant", topics=("kubernetes",),
                   novelty_links=[("experiment", "music_theory")])
    out = apply_feedback(p, entry, alpha=0.3)
    # the novel topic, previously absent, is now an affinity
    assert "music_theory" not in p.affinities.topics
    assert out.affinities.topics.get("music_theory", 0) > 0


# ---- T-P7: (3, any) produces a small/zero delta ----


@pytest.mark.unit
def test_p7_lukewarm_small_delta():
    p = _profile()
    out_relevant = apply_feedback(copy.deepcopy(p), _entry(5, "relevant"), alpha=0.3)
    out_luke = apply_feedback(copy.deepcopy(p), _entry(3, "relevant"), alpha=0.3)
    luke = out_luke.affinities.topics.get("kubernetes", 0.0)
    strong = out_relevant.affinities.topics.get("kubernetes", 0.0)
    assert luke < strong
    assert luke <= 0.05  # essentially negligible


# ---- T-P8: updates are EMA-bounded — one event cannot move a weight past a cap ----


@pytest.mark.unit
def test_p8_ema_bounded_single_event():
    p = _profile()
    out = apply_feedback(p, _entry(5, "relevant"), alpha=0.3)
    w = out.affinities.topics["kubernetes"]
    assert 0.0 < w <= 0.5  # with alpha=0.3 toward target 1.0, a single step is bounded
    assert all(v <= 1.0 for v in out.affinities.topics.values())


@pytest.mark.unit
def test_confidence_rises_with_documents():
    p = _profile()
    out = apply_feedback(p, _entry(5, "relevant"), alpha=0.3)
    assert out.meta.confidence >= p.meta.confidence


@pytest.mark.unit
def test_repeated_relevant_converges_not_overshoots():
    p = _profile()
    for _ in range(50):
        p = apply_feedback(p, _entry(5, "relevant"), alpha=0.3)
    assert p.affinities.topics["kubernetes"] <= 1.0
    assert p.affinities.topics["kubernetes"] > 0.8  # converges toward target


@pytest.mark.unit
def test_pure_does_not_mutate_input():
    p = _profile()
    before = p.model_dump_json()
    apply_feedback(p, _entry(5, "relevant"), alpha=0.3)
    assert p.model_dump_json() == before
