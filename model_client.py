"""Reusable model client for the Employee Onboarding AI Agent.

This module isolates all Google Gemini (google-genai) provider code from the
FastAPI layer. The rest of the application depends only on:

    generate_response(message: str) -> str

and on the custom exception hierarchy defined below. No raw provider response
objects and no secrets ever cross this boundary.
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

logger = logging.getLogger("model_client")

# Name of the environment variable holding the provider API key.
API_KEY_ENV = "API_key"

# Default Gemini model. Kept small/fast for an educational project.
# "gemini-flash-latest" is a stable alias that stays available as versions roll.
DEFAULT_MODEL = "gemini-flash-latest"


# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #
class ModelClientError(Exception):
    """Base class for all model-client errors."""


class ConfigurationError(ModelClientError):
    """The client is misconfigured (e.g. missing API key)."""


class AuthenticationError(ModelClientError):
    """The provider rejected our credentials (invalid API key / access denied)."""


class ProviderRequestError(ModelClientError):
    """The provider request failed (network error, server error, etc.)."""


class EmptyModelResponseError(ModelClientError):
    """The provider returned no usable text."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _get_api_key() -> str:
    """Return the configured API key or raise ConfigurationError.

    The key value itself is never logged or included in error messages.
    """
    api_key = os.getenv(API_KEY_ENV)
    if not api_key or not api_key.strip():
        raise ConfigurationError(
            f"Missing API key: environment variable '{API_KEY_ENV}' is not set."
        )
    return api_key


def _build_client(api_key: str):
    """Create a google-genai client.

    Imported lazily so that importing this module (e.g. during tests that only
    exercise validation) does not require the SDK to be fully initialised.
    """
    from google import genai

    return genai.Client(api_key=api_key)


def _is_auth_error(exc: Exception) -> bool:
    """Best-effort detection of authentication/permission failures."""
    text = str(exc).lower()
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return any(
        marker in text
        for marker in (
            "api key",
            "api_key",
            "unauthenticated",
            "permission",
            "invalid authentication",
            "401",
            "403",
        )
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_response(message: str, model: str = DEFAULT_MODEL) -> str:
    """Send ``message`` to the LLM provider and return the generated text.

    Args:
        message: The user prompt. Must be a non-empty, non-whitespace string.
        model: The provider model name to use.

    Returns:
        The generated answer as a clean Python string.

    Raises:
        ConfigurationError: API key is missing.
        AuthenticationError: Provider rejected the credentials.
        ProviderRequestError: The provider request failed.
        EmptyModelResponseError: The provider returned no usable text.
    """
    # Basic precondition validation (defence in depth; the API layer also validates).
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty, non-whitespace string.")

    api_key = _get_api_key()
    client = _build_client(api_key)

    # Log only the message length, never the message content or the key.
    logger.info("Model request started (model=%s, message_len=%d)", model, len(message))
    start = time.perf_counter()

    try:
        response = client.models.generate_content(model=model, contents=message)
    except Exception as exc:  # noqa: BLE001 - normalise SDK errors into our hierarchy
        latency_ms = (time.perf_counter() - start) * 1000
        if _is_auth_error(exc):
            logger.error(
                "Model request failed: authentication error (latency=%.1f ms)",
                latency_ms,
            )
            raise AuthenticationError("Provider authentication failed.") from exc
        logger.error(
            "Model request failed: provider error (latency=%.1f ms)", latency_ms
        )
        raise ProviderRequestError("Provider request failed.") from exc

    latency_ms = (time.perf_counter() - start) * 1000

    # Extract text safely; do not return the raw provider object.
    text = getattr(response, "text", None)
    if not text or not text.strip():
        logger.error(
            "Model request returned empty response (latency=%.1f ms)", latency_ms
        )
        raise EmptyModelResponseError("Provider returned an empty response.")

    logger.info("Model request completed (latency=%.1f ms)", latency_ms)
    return text.strip()
