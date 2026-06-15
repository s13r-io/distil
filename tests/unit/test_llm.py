"""Phase 0.2 — LLMClient protocol, FakeClient, AnthropicClient skeleton.

The LLM boundary is the seam that keeps the pipeline testable (ARCHITECTURE.md §5):
deterministic glue is unit-tested against a FakeClient returning canned responses; real
model behaviour is exercised only in the gated eval suite.
"""

import pytest

from distil.llm import AnthropicClient, FakeClient, LLMClient


@pytest.mark.unit
def test_fakeclient_returns_canned_response_in_order():
    fake = FakeClient(responses=["first", "second"])
    assert fake.complete("prompt A") == "first"
    assert fake.complete("prompt B") == "second"


@pytest.mark.unit
def test_fakeclient_records_calls_for_assertions():
    fake = FakeClient(responses=["ok"])
    fake.complete("hello", system="be terse")
    assert fake.call_count == 1
    assert fake.calls[0].prompt == "hello"
    assert fake.calls[0].system == "be terse"


@pytest.mark.unit
def test_fakeclient_raises_when_exhausted():
    fake = FakeClient(responses=["only one"])
    fake.complete("first")
    with pytest.raises(IndexError):
        fake.complete("second")


@pytest.mark.unit
def test_fakeclient_zero_calls_by_default():
    # Used by the abstention test (T-Q2): assert the answer method was never called.
    fake = FakeClient(responses=["unused"])
    assert fake.call_count == 0


@pytest.mark.unit
def test_fakeclient_satisfies_protocol():
    assert isinstance(FakeClient(responses=[]), LLMClient)


@pytest.mark.unit
def test_anthropic_client_reads_model_and_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    client = AnthropicClient()
    assert client.model == "claude-test-model"
    assert isinstance(client, LLMClient)


@pytest.mark.unit
def test_anthropic_client_missing_key_is_friendly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    client = AnthropicClient()
    # Construction must not require the SDK or a network; calling without a key is the error.
    with pytest.raises(RuntimeError) as exc:
        client.complete("hi")
    assert "ANTHROPIC_API_KEY" in str(exc.value)


@pytest.mark.unit
def test_anthropic_client_missing_model_is_friendly(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("DISTIL_MODEL", raising=False)
    with pytest.raises(RuntimeError) as exc:
        AnthropicClient()
    assert "DISTIL_MODEL" in str(exc.value)
