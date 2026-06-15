"""Phase 10.1 — Embedder protocol + FakeEmbedder. Test T-X3 (pluggability)."""

import pytest

from distil.embed import Embedder, FakeEmbedder


@pytest.mark.unit
def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=8)
    v1 = e.embed("hello world")
    v2 = e.embed("hello world")
    assert v1 == v2
    assert len(v1) == 8


@pytest.mark.unit
def test_fake_embedder_distinguishes_texts():
    e = FakeEmbedder(dim=16)
    assert e.embed("kubernetes networking") != e.embed("baking sourdough bread")


@pytest.mark.unit
def test_fake_embedder_similar_texts_closer_than_dissimilar():
    from distil.query import cosine

    e = FakeEmbedder(dim=64)
    base = e.embed("how to write unit tests in python")
    near = e.embed("writing python unit tests")
    far = e.embed("the history of medieval france")
    assert cosine(base, near) > cosine(base, far)


@pytest.mark.unit
def test_embed_batch():
    e = FakeEmbedder(dim=8)
    vecs = e.embed_batch(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 8 for v in vecs)


@pytest.mark.unit
def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(dim=4), Embedder)


@pytest.mark.unit
def test_model_name_exposed_for_reindex_consistency():
    e = FakeEmbedder(dim=8)
    assert e.model_name  # non-empty identifier stored alongside vectors
