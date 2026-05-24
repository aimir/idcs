"""Tests for the LLM call-budget tracker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from idcs.llm import LLM, BudgetExceededError


def _fake_client(text: str = "ok") -> MagicMock:
    """A minimal stand-in for openai.OpenAI() that returns deterministic text."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def test_counts_each_complete_call() -> None:
    llm = LLM(api_key="dummy")
    llm.client = _fake_client()

    assert llm.calls_made == 0
    llm.complete("sys", "u1")
    assert llm.calls_made == 1
    llm.complete("sys", "u2")
    assert llm.calls_made == 2


def test_budget_blocks_when_exhausted() -> None:
    llm = LLM(api_key="dummy", max_calls=2)
    llm.client = _fake_client()

    llm.complete("sys", "u1")
    llm.complete("sys", "u2")
    with pytest.raises(BudgetExceededError):
        llm.complete("sys", "u3")
    # Budget refused the third call before it left the process.
    assert llm.calls_made == 2


def test_unset_budget_is_unlimited() -> None:
    llm = LLM(api_key="dummy")  # default max_calls=None
    llm.client = _fake_client()

    for _ in range(20):
        llm.complete("sys", "u")
    assert llm.calls_made == 20  # nothing raised
