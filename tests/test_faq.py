"""Tests for the prompt-only HR FAQ logic (provider mocked, no real API calls).

These tests verify that answer_question builds a grounded prompt and returns the
expected structured shape. They do NOT call the real model. The six required
scenarios (direct, paraphrase, remote work, working hours, unknown, false
assumption) are exercised against a fake model that echoes the prompt, so we can
assert the policy grounding and fallback instructions are present.
"""

from __future__ import annotations

import pytest

import hr_faq
import model_client


def test_answer_question_shape(monkeypatch):
    monkeypatch.setattr(model_client, "generate_response", lambda prompt: "some answer")
    result = hr_faq.answer_question("How many annual leave days?")
    assert result == {"answer": "some answer", "mode": "prompt_only_faq"}


def test_answer_question_rejects_blank():
    with pytest.raises(ValueError):
        hr_faq.answer_question("   ")


def test_prompt_contains_policy_and_rules():
    prompt = hr_faq.build_prompt("Can I work from home?")
    lower = prompt.lower()
    # Core policy facts must be present so the model is grounded.
    assert "20 working days" in lower
    assert "2 days per week" in lower
    assert "09:00 to 17:00" in lower
    assert "before 09:00" in lower
    # Grounding rules must be present.
    assert hr_faq.FALLBACK_ANSWER in prompt
    assert "do not use external" in lower
    # The user's question must be included.
    assert "Can I work from home?" in prompt


# --------------------------------------------------------------------------- #
# The six required scenarios (fake model echoes the prompt back).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "How many annual leave days do employees receive?",   # direct
        "What is the yearly vacation allowance?",             # paraphrase
        "Can I work from home?",                              # remote work
        "When does the normal workday start and end?",        # working hours
        "Does the company provide a free car?",               # unknown
        "Employees can work remotely five days per week, correct?",  # false assumption
    ],
)
def test_required_scenarios_are_grounded(monkeypatch, question):
    # Echo the prompt so we can confirm the policy + question reached the model.
    monkeypatch.setattr(model_client, "generate_response", lambda prompt: prompt)
    result = hr_faq.answer_question(question)
    assert result["mode"] == "prompt_only_faq"
    assert question in result["answer"]
    assert "HR POLICY" in result["answer"]
