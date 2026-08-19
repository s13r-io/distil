"""Deterministic tests for the two-pass prose style rewrite."""

import json

import pytest

from distil.llm import FakeClient
from distil.unslop import rewrite_json_fields, rewrite_text


@pytest.mark.unit
def test_rewrite_text_runs_rewrite_then_self_audit_with_the_raw_guide():
    client = FakeClient(["Plain first pass.", "Plain final pass."])

    result = rewrite_text("Additionally, this is pivotal.", client)

    assert result == "Plain final pass."
    assert client.call_count == 2
    assert "# Unslop" in client.calls[0].prompt
    assert "Additionally, this is pivotal." in client.calls[0].prompt
    assert "Plain first pass." in client.calls[1].prompt
    assert "What in this still reads as obviously AI-written?" in client.calls[1].prompt
    for call in client.calls:
        assert "preserve meaning and facts exactly" in call.prompt.lower()
        assert "never invent" in call.prompt.lower()


@pytest.mark.unit
def test_rewrite_text_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("DISTIL_UNSLOP_ENABLED", raising=False)
    client = FakeClient(["first", "final"])

    assert rewrite_text("original", client) == "final"
    assert client.call_count == 2


@pytest.mark.unit
def test_rewrite_text_env_toggle_disables_both_calls(monkeypatch):
    monkeypatch.setenv("DISTIL_UNSLOP_ENABLED", "false")
    client = FakeClient([])

    assert rewrite_text("original", client) == "original"
    assert client.call_count == 0


@pytest.mark.unit
def test_rewrite_text_rejects_a_non_summary_tier_client(monkeypatch):
    monkeypatch.setenv("DISTIL_MODEL_SUMMARY", "cheap-summary-model")
    client = FakeClient(["first", "final"])
    client.model = "strong-note-model"

    assert rewrite_text("original", client) == "original"
    assert client.call_count == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "responses",
    [
        [ConnectionError("first pass failed")],
        ["first", TimeoutError("audit timed out")],
    ],
)
def test_rewrite_text_failure_falls_back_without_raising(responses):
    assert rewrite_text("original", FakeClient(responses)) == "original"


@pytest.mark.unit
def test_rewrite_json_fields_rejects_changed_citation_ids_and_falls_back():
    original = {
        "key_points": [{"text": "Additionally, use the check.", "item_ids": ["k_01"]}],
        "how_to_apply": [
            {
                "text": "Leverage the checklist.",
                "item_ids": ["k_01"],
                "application_link_ids": ["a_01"],
            }
        ],
    }
    corrupted = {
        "key_points": [{"text": "Use the check.", "item_ids": ["k_02"]}],
        "how_to_apply": [
            {
                "text": "Use the checklist.",
                "item_ids": ["k_01"],
                "application_link_ids": ["a_01"],
            }
        ],
    }
    client = FakeClient([json.dumps(original), json.dumps(corrupted)])

    result = rewrite_json_fields(
        original,
        client,
        text_keys={"text"},
        id_keys={"item_ids", "application_link_ids"},
    )

    assert result == original


@pytest.mark.unit
def test_rewrite_json_fields_batches_all_text_and_accepts_only_text_changes():
    original = [
        {"link_id": "a_01", "knowledge_item_ids": ["k_01"], "scenario": "Additionally do X."},
        {"link_id": "a_02", "knowledge_item_ids": ["k_02"], "scenario": "Crucially do Y."},
    ]
    rewritten = [
        {"link_id": "a_01", "knowledge_item_ids": ["k_01"], "scenario": "Do X."},
        {"link_id": "a_02", "knowledge_item_ids": ["k_02"], "scenario": "Do Y."},
    ]
    client = FakeClient([json.dumps(rewritten), json.dumps(rewritten)])

    result = rewrite_json_fields(
        original,
        client,
        text_keys={"scenario"},
        id_keys={"knowledge_item_ids"},
    )

    assert result == rewritten
    assert client.call_count == 2
