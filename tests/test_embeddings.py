"""Tests for the embedding experiment utilities (no real API calls).

Covers the cosine-similarity utility and the ranking/embedding helpers, with the
embedding API mocked so tests are deterministic and consume no quota.
"""

from __future__ import annotations

import math
import types

import pytest

import embedding_experiment as ee
import model_client
from similarity import cosine_similarity


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #
def test_identical_vectors_similarity_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_zero_vector_does_not_crash():
    # A zero-length vector has no direction; defined as 0.0, not a crash.
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_vector_raises():
    with pytest.raises(ValueError):
        cosine_similarity([], [1.0])


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_ranking_sorted_high_to_low():
    # query = [1,0]; nearest is [1,0], then [1,1], then [0,1].
    vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ranked = ee.rank_by_similarity(vectors, query_index=0)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    # Query itself is excluded.
    assert all(idx != 0 for idx, _ in ranked)
    # The exact-duplicate vector ranks first.
    assert ranked[0][0] == 1


# --------------------------------------------------------------------------- #
# embed_sentences (embedding API mocked)
# --------------------------------------------------------------------------- #
def _fake_client(vectors=None, raise_exc=None, embeddings=None):
    def embed_content(model, contents):
        if raise_exc is not None:
            raise raise_exc
        if embeddings is not None:
            return types.SimpleNamespace(embeddings=embeddings)
        items = [types.SimpleNamespace(values=v) for v in vectors]
        return types.SimpleNamespace(embeddings=items)

    models = types.SimpleNamespace(embed_content=embed_content)
    return types.SimpleNamespace(models=models)


def test_embed_sentences_rejects_empty_sentence(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    with pytest.raises(ValueError):
        ee.embed_sentences(["ok", "   "])


def test_embed_sentences_returns_vectors(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client",
        lambda key: _fake_client(vectors=[[1.0, 2.0], [3.0, 4.0]]),
    )
    result = ee.embed_sentences(["a", "b"])
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_sentences_dimension_mismatch_raises(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client",
        lambda key: _fake_client(vectors=[[1.0, 2.0], [3.0]]),
    )
    with pytest.raises(ValueError):
        ee.embed_sentences(["a", "b"])


def test_embed_sentences_empty_response_raises(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _fake_client(embeddings=[])
    )
    with pytest.raises(model_client.EmptyModelResponseError):
        ee.embed_sentences(["a", "b"])


def test_embed_sentences_api_failure_mapped(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client",
        lambda key: _fake_client(raise_exc=Exception("network down")),
    )
    with pytest.raises(model_client.ProviderRequestError):
        ee.embed_sentences(["a", "b"])
