# Employee Onboarding AI Assistant

A backend service that answers new-employee onboarding questions in natural language.
It exposes a clean REST API built with **FastAPI** and generates answers using
**Google Gemini** through the official `google-genai` SDK.

The service is designed to be the foundation of an internal HR assistant: a new hire
(or a frontend/chat client) sends a question, the API validates it, forwards it to the
language model behind a reusable client, and returns a structured JSON answer.

---

## Features

- **REST API** with automatic, interactive documentation (Swagger UI + OpenAPI schema).
- **Health endpoint** (`GET /health`) for uptime and readiness checks.
- **Chat endpoint** (`POST /chat`) that returns model-generated answers as validated JSON.
- **Strict request validation** with Pydantic (type, length, and non-blank checks).
- **Reusable model client** that isolates all provider/SDK code behind a single function.
- **Safe error handling** — internal failures are mapped to appropriate HTTP status codes
  with generic messages; secrets, stack traces, and provider internals are never exposed.
- **Structured logging** of request lifecycle and latency (without leaking sensitive data).
- **Automated test suite** covering the API and the model client, with the provider mocked
  so tests are fast, deterministic, and free of API usage.

---

## Architecture

```
Client / Swagger UI
   → JSON request
   → Pydantic validation
   → FastAPI route (POST /chat)
   → reusable model client
   → Google Gemini API
   → extracted answer text
   → Pydantic response model
   → JSON response
```

The API key lives only on the backend (in a local, git-ignored `.env` file) and is read
exclusively inside the model client. It never appears in requests, responses, logs, or
version control.

---

## Tech stack

| Layer            | Technology                     |
|------------------|--------------------------------|
| Web framework    | FastAPI                        |
| Server           | Uvicorn                        |
| Validation       | Pydantic v2                    |
| LLM provider     | Google Gemini (`google-genai`) |
| Config           | python-dotenv                  |
| Testing          | pytest                         |

---

## Getting started

### 1. Requirements

- Python 3.12+
- A Google Gemini API key (from Google AI Studio)

### 2. Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure the API key

Copy the example environment file and add your key:

```powershell
copy .env.example .env
```

Then edit `.env`:

```text
API_key=your_gemini_api_key_here
```

`.env` is git-ignored and must never be committed.

### 4. Run the API

```powershell
uvicorn main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

---

## API reference

### `GET /health`

Returns service status.

```json
{ "status": "ok" }
```

### `POST /chat`

**Request**

```json
{ "message": "What is the remote work policy?" }
```

**Response**

```json
{ "answer": "…model-generated answer…", "status": "success" }
```

**Validation rules for `message`:** required, string, 1–2000 characters, not blank/whitespace-only.
Invalid input returns `422`. Provider or configuration failures return safe `5xx` responses.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"How do I set up my work laptop?\"}"
```

---

## Testing

```powershell
pytest -q
```

Tests mock the language-model provider, so they run offline, consume no API quota, and
deterministically cover both success and failure paths (validation errors, missing
configuration, authentication failure, provider errors, empty responses, and unexpected
internal errors).

---

## Error handling

| Situation                          | HTTP status |
|------------------------------------|-------------|
| Invalid / blank / oversized input  | 422 (or 400) |
| Missing server configuration       | 503         |
| Provider authentication failure    | 503         |
| Provider request failure           | 502         |
| Empty provider response            | 502         |
| Unexpected internal error          | 500         |

Responses contain only safe, user-facing messages. Detailed diagnostics (including
tracebacks) are logged internally and never returned to the client.

---

## Project structure

```
.
├── main.py                 # FastAPI app: routes, request/response models, error mapping
├── model_client.py         # Reusable Gemini client, error hierarchy, logging, latency
├── first_call.py           # Minimal standalone example of a single model call
├── requirements.txt        # Dependencies
├── .env.example            # Environment variable template (placeholder only)
├── tests/
│   ├── test_api.py         # API endpoint tests
│   └── test_model_client.py# Model client unit tests
└── README.md
```

---

## Roadmap

Planned enhancements to grow this into a full internal assistant:

- Retrieval-augmented answers grounded in company policy documents.
- Conversation memory for multi-turn support.
- Tool/action support (e.g. creating HR support tickets).
- A simple web chat interface.

---

## License

Provided for internal and educational use.
