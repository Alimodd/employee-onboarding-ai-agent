"""API tests for the FastAPI application (deterministic, no real API calls)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
import model_client

client = TestClient(main.app)


# --------------------------------------------------------------------------- #
# Health + routing + schema
# --------------------------------------------------------------------------- #
def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_route_404():
    assert client.get("/does-not-exist").status_code == 404


def test_openapi_available():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/health" in schema["paths"]
    assert "/chat" in schema["paths"]


def test_docs_available():
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------------------- #
# /chat validation (Pydantic 422)
# --------------------------------------------------------------------------- #
def test_chat_missing_message_422():
    assert client.post("/chat", json={}).status_code == 422


def test_chat_wrong_type_422():
    assert client.post("/chat", json={"message": 123}).status_code == 422


def test_chat_empty_string_422():
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_whitespace_only_422():
    assert client.post("/chat", json={"message": "    "}).status_code == 422


def test_chat_oversized_422():
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422


# --------------------------------------------------------------------------- #
# /chat success + failure flows (model client patched)
# --------------------------------------------------------------------------- #
def test_chat_success(monkeypatch):
    monkeypatch.setattr(
        model_client, "generate_response", lambda msg: f"echo::{msg}"
    )
    resp = client.post("/chat", json={"message": "What is the remote work policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["answer"] == "echo::What is the remote work policy?"


def test_chat_success_not_hardcoded(monkeypatch):
    monkeypatch.setattr(model_client, "generate_response", lambda msg: msg.upper())
    a = client.post("/chat", json={"message": "hello"}).json()["answer"]
    b = client.post("/chat", json={"message": "world"}).json()["answer"]
    assert a == "HELLO" and b == "WORLD"


def test_chat_configuration_error_503(monkeypatch):
    def boom(msg):
        raise model_client.ConfigurationError("missing key")

    monkeypatch.setattr(model_client, "generate_response", boom)
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_auth_error_503(monkeypatch):
    def boom(msg):
        raise model_client.AuthenticationError("bad key")

    monkeypatch.setattr(model_client, "generate_response", boom)
    assert client.post("/chat", json={"message": "hi"}).status_code == 503


def test_chat_empty_response_502(monkeypatch):
    def boom(msg):
        raise model_client.EmptyModelResponseError("empty")

    monkeypatch.setattr(model_client, "generate_response", boom)
    assert client.post("/chat", json={"message": "hi"}).status_code == 502


def test_chat_provider_error_502(monkeypatch):
    def boom(msg):
        raise model_client.ProviderRequestError("down")

    monkeypatch.setattr(model_client, "generate_response", boom)
    assert client.post("/chat", json={"message": "hi"}).status_code == 502


def test_chat_unexpected_error_500(monkeypatch):
    def boom(msg):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(model_client, "generate_response", boom)
    assert client.post("/chat", json={"message": "hi"}).status_code == 500


def test_chat_no_secret_in_error(monkeypatch):
    def boom(msg):
        raise model_client.AuthenticationError("bad key")

    monkeypatch.setattr(model_client, "generate_response", boom)
    resp = client.post("/chat", json={"message": "hi"})
    assert "key" not in resp.text.lower() or "api" not in resp.text.lower()
