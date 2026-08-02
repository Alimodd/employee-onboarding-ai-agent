"""FastAPI application for the Employee Onboarding AI Agent (Days 1-6).

Endpoints:
    GET  /health  -> {"status": "ok"}
    POST /chat    -> {"answer": "...", "status": "success"}

The /chat route validates input with Pydantic, delegates to the reusable model
client, and translates internal errors into safe HTTP responses. The provider
SDK and the API key live exclusively behind model_client; they never appear in
this layer or in any response.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

import model_client

# Basic logging configuration for an educational project.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Employee Onboarding AI Agent",
    description="Educational FastAPI backend (Days 1-6): health check and LLM chat.",
    version="0.6.0",
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question for the onboarding assistant.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        # Reject whitespace-only input even though Pydantic accepts it as a string.
        if not value.strip():
            raise ValueError("message must not be empty or whitespace-only.")
        return value


class ChatResponse(BaseModel):
    answer: str
    status: str = "success"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Proves the app is running and can serve requests."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Validate the request, call the model client, return a validated answer."""
    logger.info("Chat request received (message_len=%d)", len(request.message))

    try:
        answer = model_client.generate_response(request.message)
    except model_client.ConfigurationError:
        # Server is misconfigured (e.g. missing API key). Do not leak details.
        logger.exception("Chat failed: server configuration error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not configured correctly. Please try again later.",
        )
    except model_client.AuthenticationError:
        logger.exception("Chat failed: provider authentication error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upstream service is unavailable. Please try again later.",
        )
    except model_client.EmptyModelResponseError:
        logger.exception("Chat failed: empty provider response")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model did not return a response. Please try again.",
        )
    except model_client.ProviderRequestError:
        logger.exception("Chat failed: provider request error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model provider could not be reached. Please try again.",
        )
    except ValueError:
        # Defensive: precondition failure from the model client.
        logger.exception("Chat failed: invalid input reached model client")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input.",
        )
    except Exception:  # noqa: BLE001 - last-resort guard; return a generic error.
        logger.exception("Chat failed: unexpected internal error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
        )

    logger.info("Chat request completed successfully")
    return ChatResponse(answer=answer, status="success")
