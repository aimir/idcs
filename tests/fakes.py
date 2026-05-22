"""Test doubles for ``LLMClient`` and ``UserProxy``.

Kept in ``tests/`` because they exist only to drive structural tests of the
orchestrator without making real API calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

TypedResponder = Callable[[str, str, type[BaseModel]], BaseModel]
TextResponder = Callable[[str, str], str]


@dataclass
class FakeLLM:
    """LLM client that replays queued responses.

    Each ``complete_typed`` call pops from ``typed_responses``; each
    ``complete`` call pops from ``text_responses``. Use ``typed_responder``
    or ``text_responder`` for stateful behavior keyed off the user message.
    """

    typed_responses: list[BaseModel] = field(default_factory=list)
    text_responses: list[str] = field(default_factory=list)
    typed_responder: TypedResponder | None = None
    text_responder: TextResponder | None = None

    typed_calls: list[tuple[str, str, type[BaseModel]]] = field(default_factory=list)
    text_calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 16000,
        thinking: bool = False,
        cache_system: bool = True,
    ) -> str:
        self.text_calls.append((system, user))
        if self.text_responder is not None:
            return self.text_responder(system, user)
        if not self.text_responses:
            raise AssertionError("FakeLLM: no text response queued")
        return self.text_responses.pop(0)

    def complete_typed(
        self,
        system: str,
        user: str,
        output_type: type[T],
        *,
        max_tokens: int = 16000,
        thinking: bool = False,
        cache_system: bool = True,
    ) -> T:
        self.typed_calls.append((system, user, output_type))
        if self.typed_responder is not None:
            result = self.typed_responder(system, user, output_type)
            if not isinstance(result, output_type):
                raise AssertionError(
                    f"FakeLLM typed_responder returned {type(result).__name__}, "
                    f"expected {output_type.__name__}"
                )
            return result
        if not self.typed_responses:
            raise AssertionError(f"FakeLLM: no typed response queued for {output_type.__name__}")
        result = self.typed_responses.pop(0)
        if not isinstance(result, output_type):
            raise AssertionError(
                f"FakeLLM: queued response is {type(result).__name__}, "
                f"expected {output_type.__name__}"
            )
        return result


@dataclass
class FakeUserProxy:
    """User proxy that replays answers keyed by issue location."""

    answers: dict[str, str | None] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def answer(self, location: str, question: str) -> str | None:
        self.calls.append((location, question))
        return self.answers.get(location)
