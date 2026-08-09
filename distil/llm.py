"""LLM boundary for Distil.

Every LLM-backed stage takes an injected :class:`LLMClient`, so deterministic glue can be
unit-tested against :class:`FakeClient` while real model behaviour is exercised only in the
gated eval suite (ARCHITECTURE.md §5, TESTING.md §1).

The provider is swappable: implement :class:`LLMClient` and inject it. The default provider
is Anthropic / Claude, reading its key and model from the environment (no hardcoded model
string; ARCHITECTURE.md §1).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-completion interface every provider must satisfy."""

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's text completion for ``prompt`` (optionally guided by ``system``)."""
        ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        """Yield text deltas for ``prompt``. Providers without native streaming may yield once.

        Optional in practice: callers should fall back to :meth:`complete` if a client doesn't
        implement streaming (WEB_UI_SPEC §9 safety valve).
        """
        ...


@dataclass
class _Call:
    prompt: str
    system: str | None = None


class FakeClient:
    """Deterministic test double returning canned responses in order.

    Records every call so tests can assert *that* the model was (or was not) invoked —
    critical for the abstention guarantee (T-Q2), which asserts zero synthesis calls.
    """

    def __init__(self, responses: list[str | Exception] | None = None):
        self._responses = list(responses or [])
        self._cursor = 0
        self.calls: list[_Call] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return (or raise) the next canned response.

        A queued :class:`Exception` instance is raised instead of returned, so tests can
        simulate a dropped connection ahead of a later successful retry.
        """
        self.calls.append(_Call(prompt=prompt, system=system))
        if self._cursor >= len(self._responses):
            raise IndexError("FakeClient ran out of canned responses")
        response = self._responses[self._cursor]
        self._cursor += 1
        if isinstance(response, BaseException):
            raise response
        return response

    def stream(self, prompt: str, *, system: str | None = None):
        """Yield the next canned response in word-sized chunks (records the call like complete)."""
        response = self.complete(prompt, system=system)
        words = response.split(" ")
        for i, w in enumerate(words):
            yield (w if i == 0 else " " + w)

    @property
    def call_count(self) -> int:
        return len(self.calls)


# A non-streaming request whose max_tokens estimates past ~10 minutes of generation is rejected
# client-side by the SDK (ValueError) before it ever reaches the network — and even where the SDK
# would allow it, a long-held non-streaming connection risks an idle-connection drop server- or
# proxy-side. Above this threshold, complete() routes through the streaming API internally and
# assembles the final text itself, so callers keep the same synchronous str-returning interface.
# See the PR that introduced per-stage max_tokens ceilings (distil/model_config.py) for why
# extraction alone needs a ceiling this high.
_STREAMING_REQUIRED_ABOVE_TOKENS = 16_000


@dataclass
class AnthropicClient:
    """Default provider skeleton: Claude API, key + model from env (ARCHITECTURE.md §1, §6).

    The model is required at construction; the API key is required only when a completion is
    actually requested, so the object can be constructed (and a friendly missing-key error
    surfaced lazily) without a network or the SDK installed.

    ``max_tokens`` is the per-request output *ceiling*, not a spend — the API only bills for
    tokens actually generated, so a caller raising this costs nothing unless the model produces
    more output. It defaults to 4096 (the pre-existing hardcoded value) for any caller that
    doesn't specify one explicitly; ``distil/model_config.py``'s ``make_stage_client`` is what
    resolves the right ceiling per pipeline stage.
    """

    model: str = field(default="")
    max_tokens: int = field(default=4096)
    _api_key: str | None = field(default=None, repr=False)

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        resolved_model = model or os.environ.get("DISTIL_MODEL", "")
        if not resolved_model:
            raise RuntimeError(
                "DISTIL_MODEL is not set. Set it in your .env to a current model string "
                "(see .env.example) — Distil does not hardcode a model."
            )
        self.model = resolved_model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env (see .env.example) to use "
                "the Anthropic provider, or inject a different LLMClient."
            )
        try:
            import anthropic  # imported lazily so the package isn't a hard dependency
        except ImportError as exc:  # pragma: no cover - exercised only with provider installed
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install it with "
                "`pip install distil[anthropic]`."
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens,
                        "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        if self.max_tokens > _STREAMING_REQUIRED_ABOVE_TOKENS:
            with client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        else:
            message = client.messages.create(**kwargs)
        return "".join(block.text for block in message.content if block.type == "text")

    def stream(self, prompt: str, *, system: str | None = None):  # pragma: no cover - network
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env (see .env.example) to use "
                "the Anthropic provider, or inject a different LLMClient."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install it with "
                "`pip install distil[anthropic]`."
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens,
                        "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream
