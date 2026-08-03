"""Prompt-only HR FAQ logic for the Employee Onboarding AI Assistant.

This module answers HR questions using a small, hard-coded, fictional HR policy
that is injected directly into the model prompt. There is no document retrieval,
no embeddings, and no vector database here: the entire "knowledge" is the string
constant ``HR_POLICY`` below, handed to the model as grounding context.

The FastAPI layer calls :func:`answer_question`. The actual provider/API call is
delegated to ``model_client.generate_response`` so that API-key loading and SDK
handling are never duplicated.
"""

from __future__ import annotations

import model_client

# Identifies this answering strategy in the API response.
MODE = "prompt_only_faq"

# Exact fallback meaning required when the policy does not cover a question.
FALLBACK_ANSWER = "I could not find this information in the available HR policy."

# --------------------------------------------------------------------------- #
# Fictional, hard-coded HR policy (the model's only allowed source of truth)
# --------------------------------------------------------------------------- #
HR_POLICY = """\
Fictional HR policy for an educational demo company (Northstar Analytics).

- Annual leave: employees receive 20 working days of annual leave per calendar year.
- Remote work: employees may work remotely up to 2 days per week, with manager approval.
- Standard working hours: Monday to Friday, 09:00 to 17:00.
- Sick leave reporting: employees must notify their manager before 09:00 on the
  first day of absence.
"""

# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_SYSTEM_INSTRUCTIONS = f"""\
You are an HR FAQ assistant. Answer strictly and only using the fictional HR
policy provided below. Follow these rules exactly:

- Use ONLY the provided HR policy. Do not use external or general knowledge.
- Do not invent company rules or add details that are not in the policy.
- If the answer is not contained in the policy, reply with exactly:
  "{FALLBACK_ANSWER}"
- If the user states something that contradicts the policy, do not agree.
  Correct the false assumption and state what the policy actually says.
- Keep answers short, factual, and directly based on the policy.

HR POLICY:
{HR_POLICY}
"""


def build_prompt(message: str) -> str:
    """Combine the grounding instructions/policy with the user's question.

    Args:
        message: The user's HR question (already validated by the caller).

    Returns:
        A single prompt string ready to send to ``model_client``.
    """
    return f"{_SYSTEM_INSTRUCTIONS}\nUSER QUESTION:\n{message}\n\nANSWER:"


def answer_question(message: str) -> dict:
    """Answer an HR question using the hard-coded policy prompt.

    Args:
        message: The user's question. Must be a non-empty, non-whitespace string.

    Returns:
        A dict of the form ``{"answer": <str>, "mode": "prompt_only_faq"}``.

    Raises:
        ValueError: If ``message`` is empty or whitespace-only.
        model_client.ModelClientError: Any provider/configuration failure,
            propagated unchanged so the API layer can map it to an HTTP status.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty, non-whitespace string.")

    prompt = build_prompt(message)
    answer = model_client.generate_response(prompt)
    return {"answer": answer, "mode": MODE}
