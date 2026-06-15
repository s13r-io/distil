"""Phase 6 — link.py (unit, FakeClient). Tests T-L1, T-L2, T-L3."""

import json

import pytest

from distil.link import generate_links, valid_goal_ids
from distil.llm import FakeClient
from distil.models import KnowledgeItem, Profile


def _profile(confidence=0.5, with_focus=True) -> Profile:
    data = {
        "user_id": "owner",
        "stable": {
            "role": "engineer",
            "long_term_goals": [
                {"id": "g_01", "statement": "ship reliable services", "created_at": "2026-01-01T00:00:00"},
                {"id": "g_02", "statement": "mentor juniors", "created_at": "2026-01-01T00:00:00"},
            ],
        },
        "meta": {"documents_processed": 10, "confidence": confidence},
    }
    if with_focus:
        data["current_focus"] = [{
            "id": "f_01", "project": "auth rewrite", "description": "replace legacy auth",
            "active_since": "2026-06-01T00:00:00", "last_touched": "2026-06-10T00:00:00",
            "status": "active",
        }]
    return Profile.model_validate(data)


def _items(n=1) -> list[KnowledgeItem]:
    return [
        KnowledgeItem.model_validate({
            "item_id": f"k_{i:02d}", "type": "heuristic", "statement": f"insight {i}",
            "stance": "opinion", "provenance": {"quote": "q"},
        })
        for i in range(1, n + 1)
    ]


def _resp(links) -> str:
    return json.dumps(links)


# ---- T-L1: every link has a valid linked_goal_id pointing at a real goal/focus ----


@pytest.mark.unit
def test_l1_links_reference_real_goals():
    p = _profile()
    items = _items(1)
    resp = _resp([
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
         "application_form": "checklist", "scenario": "apply to reliability", "novelty_flag": False},
    ])
    links = generate_links(items, p, FakeClient(responses=[resp]), novelty_ratio=0.0)
    assert len(links) == 1
    assert links[0].linked_goal_id in valid_goal_ids(p)
    assert links[0].link_id  # id assigned


@pytest.mark.unit
def test_l1_links_with_invalid_goal_are_dropped():
    p = _profile()
    resp = _resp([
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "g_99",  # not a real goal
         "application_form": "checklist", "scenario": "x", "novelty_flag": False},
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "f_01",  # real focus
         "application_form": "trigger", "scenario": "y", "novelty_flag": False},
    ])
    links = generate_links(_items(1), p, FakeClient(responses=[resp]), novelty_ratio=0.0)
    assert [link.linked_goal_id for link in links] == ["f_01"]


# ---- T-L2: novelty reservation ~ 1 in 5 with ratio 0.2 ----


@pytest.mark.unit
def test_l2_novelty_reservation():
    p = _profile()
    # 10 links, none flagged by the model → code reserves ~ratio fraction as novelty
    links_in = [
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
         "application_form": "reference", "scenario": f"s{i}", "novelty_flag": False}
        for i in range(10)
    ]
    out = generate_links(_items(1), p, FakeClient(responses=[_resp(links_in)]), novelty_ratio=0.2)
    novelty = [link for link in out if link.novelty_flag]
    assert 1 <= len(novelty) <= 3  # ~2 of 10, allow rounding tolerance


@pytest.mark.unit
def test_l2_zero_ratio_reserves_none():
    p = _profile()
    links_in = [
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
         "application_form": "reference", "scenario": f"s{i}", "novelty_flag": False}
        for i in range(5)
    ]
    out = generate_links(_items(1), p, FakeClient(responses=[_resp(links_in)]), novelty_ratio=0.0)
    assert all(not link.novelty_flag for link in out)


# ---- T-L3: cold-start (confidence 0) → links reference stable.long_term_goals ----


@pytest.mark.unit
def test_l3_cold_start_uses_stable_goals_in_prompt():
    p = _profile(confidence=0.0)
    fake = FakeClient(responses=[_resp([
        {"knowledge_item_ids": ["k_01"], "linked_goal_id": "g_01",
         "application_form": "checklist", "scenario": "x", "novelty_flag": False},
    ])])
    generate_links(_items(1), p, fake, novelty_ratio=0.0)
    prompt = fake.calls[0].prompt
    # cold start: the prompt presents stable long-term goals, not learned affinities
    assert "g_01" in prompt
    assert "ship reliable services" in prompt
    assert "affinit" not in prompt.lower()


@pytest.mark.unit
def test_valid_goal_ids_includes_goals_and_focus():
    p = _profile()
    ids = valid_goal_ids(p)
    assert {"g_01", "g_02", "f_01"} <= ids
