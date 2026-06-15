"""Embeddings (read layer). ARCHITECTURE.md §1, §9; TESTING T-X3.

A pluggable :class:`Embedder` powers semantic search. Swapping ``local`` ↔ ``api`` changes
only construction, never call sites (T-X3). For hermetic tests, :class:`FakeEmbedder` produces
deterministic vectors where word overlap raises cosine similarity — enough to exercise ranking
and the abstention gate without a model.

Real backends:
* :class:`LocalEmbedder` — a sentence-transformers model (default, provider-independent).
* :class:`ApiEmbedder` — an API embedding model (for very small hosts).

Both are imported lazily so the heavy deps aren't required unless selected.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for tests. Word overlap → higher cosine."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.model_name = f"fake-bow-{dim}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class LocalEmbedder:  # pragma: no cover - requires sentence-transformers
    """Local sentence-transformers embedder (provider-independent default)."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("DISTIL_EMBED_MODEL", "all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


class ApiEmbedder:  # pragma: no cover - requires provider SDK + key
    """API-backed embedder for small hosts; selected via DISTIL_EMBEDDER=api."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("DISTIL_EMBED_MODEL", "")
        if not self.model_name:
            raise RuntimeError("DISTIL_EMBED_MODEL must be set for the API embedder.")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "ApiEmbedder is a stub: wire your provider's embeddings endpoint here."
        )


def make_embedder() -> Embedder:
    """Construct the configured embedder (DISTIL_EMBEDDER=local|api). Local is the default."""
    backend = os.environ.get("DISTIL_EMBEDDER", "local").lower()
    if backend == "api":
        return ApiEmbedder()
    return LocalEmbedder()
