"""Phase 0.2 — LLMClient protocol, FakeClient, AnthropicClient skeleton.

The LLM boundary is the seam that keeps the pipeline testable (ARCHITECTURE.md §5):
deterministic glue is unit-tested against a FakeClient returning canned responses; real
model behaviour is exercised only in the gated eval suite.
"""

import sys
from types import ModuleType, SimpleNamespace

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


# ---- max_tokens: per-request ceiling, not a spend ---------------------------------------


@pytest.mark.unit
def test_anthropic_client_default_max_tokens_is_4096(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    assert AnthropicClient().max_tokens == 4096


@pytest.mark.unit
def test_anthropic_client_accepts_an_explicit_max_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    assert AnthropicClient(max_tokens=128_000).max_tokens == 128_000


def _fake_anthropic_module(calls: list[tuple[str, dict]]) -> ModuleType:
    """A minimal stand-in for the ``anthropic`` package, recording which of
    ``messages.create`` (non-streaming) / ``messages.stream`` (streaming) was invoked and with
    what kwargs — enough to verify AnthropicClient's internal streaming-threshold routing
    without a real network call, the real SDK's request validation, or the SDK even being
    installed (``llm.py`` imports it lazily inside the function, so it's not a hard test
    dependency — install a fake module rather than requiring the real package in CI)."""

    class _FakeMessage:
        def __init__(self, text: str):
            self.content = [SimpleNamespace(type="text", text=text)]

    class _FakeStreamContext:
        def __init__(self, message):
            self._message = message

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get_final_message(self):
            return self._message

    class _FakeMessagesAPI:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return _FakeMessage("non-streamed response")

        def stream(self, **kwargs):
            calls.append(("stream", kwargs))
            return _FakeStreamContext(_FakeMessage("streamed response"))

    class _FakeAnthropic:
        def __init__(self, api_key=None):
            self.messages = _FakeMessagesAPI()

    module = ModuleType("anthropic")
    module.Anthropic = _FakeAnthropic
    return module


@pytest.mark.unit
def test_anthropic_client_complete_uses_non_streaming_below_the_threshold(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    calls: list[tuple[str, dict]] = []
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(calls))
    client = AnthropicClient(max_tokens=4096)
    assert client.complete("hi") == "non-streamed response"
    assert [kind for kind, _ in calls] == ["create"]
    assert calls[0][1]["max_tokens"] == 4096


@pytest.mark.unit
def test_anthropic_client_complete_streams_above_the_threshold(monkeypatch):
    """A large max_tokens (e.g. extraction's full-model ceiling) must route through the
    streaming API internally — a plain non-streaming call this large either gets rejected by
    the real SDK's own timeout guard or risks an idle-connection drop; see llm.py's module-level
    comment. complete()'s public str-returning interface is unaffected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_MODEL", "claude-test-model")
    calls: list[tuple[str, dict]] = []
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(calls))
    client = AnthropicClient(max_tokens=128_000)
    assert client.complete("hi") == "streamed response"
    assert [kind for kind, _ in calls] == ["stream"]
    assert calls[0][1]["max_tokens"] == 128_000
