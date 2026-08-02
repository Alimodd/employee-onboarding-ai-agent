"""Unit tests for the reusable model client (deterministic, no real API calls)."""

from __future__ import annotations

import types

import pytest

import model_client


# --------------------------------------------------------------------------- #
# Input precondition tests
# --------------------------------------------------------------------------- #
def test_empty_message_raises_value_error(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    with pytest.raises(ValueError):
        model_client.generate_response("")


def test_whitespace_message_raises_value_error(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    with pytest.raises(ValueError):
        model_client.generate_response("   \n\t ")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv(model_client.API_KEY_ENV, raising=False)
    with pytest.raises(model_client.ConfigurationError):
        model_client.generate_response("Hello")


# --------------------------------------------------------------------------- #
# Provider interaction (mocked via _build_client)
# --------------------------------------------------------------------------- #
def _make_fake_client(text=None, raise_exc=None):
    def generate_content(model, contents):
        if raise_exc is not None:
            raise raise_exc
        return types.SimpleNamespace(text=text)

    models = types.SimpleNamespace(generate_content=generate_content)
    return types.SimpleNamespace(models=models)


def test_valid_message_returns_text(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _make_fake_client(text="  Hi there  ")
    )
    result = model_client.generate_response("Say hi")
    assert result == "Hi there"


def test_empty_provider_response_raises(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _make_fake_client(text="   ")
    )
    with pytest.raises(model_client.EmptyModelResponseError):
        model_client.generate_response("Say something")


def test_authentication_error_mapped(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "bad-key")
    exc = Exception("401 API key not valid. Please pass a valid API key.")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _make_fake_client(raise_exc=exc)
    )
    with pytest.raises(model_client.AuthenticationError):
        model_client.generate_response("Hello")


def test_provider_error_mapped(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    exc = Exception("connection reset by peer")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _make_fake_client(raise_exc=exc)
    )
    with pytest.raises(model_client.ProviderRequestError):
        model_client.generate_response("Hello")
