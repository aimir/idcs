"""Thin OpenAI-SDK wrapper pointed at OpenRouter.

OpenRouter exposes an OpenAI-compatible API and routes to many providers
(Anthropic, OpenAI, Google, Meta, ...). Switching providers is a model-id
change rather than a code change, which keeps the rest of the pipeline
single-shape.

Defaults to ``anthropic/claude-sonnet-4.5`` since the project is shaped
around Claude-style structured outputs, but any model that supports
JSON-schema ``response_format`` will work — override via the ``model``
constructor arg or the ``IDCS_MODEL`` env var.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar

import openai
from pydantic import BaseModel

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Structural type the rest of the pipeline depends on.

    The concrete ``LLM`` below satisfies this. Tests pass in a fake.
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = ...,
    ) -> str: ...

    def complete_typed(
        self,
        system: str,
        user: str,
        output_type: type[T],
        *,
        max_tokens: int = ...,
    ) -> T: ...


class LLM:
    """One-shot completion wrapper.

    Uses OpenAI's ``chat.completions`` API (the format OpenRouter exposes).
    Structured output goes through ``beta.chat.completions.parse``, which
    serializes a Pydantic model to JSON schema and validates the response.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )
        self.model = model or os.environ.get("IDCS_MODEL") or DEFAULT_MODEL

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 16000,
    ) -> str:
        """Return the assistant message's text content (empty string if absent)."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def complete_typed(
        self,
        system: str,
        user: str,
        output_type: type[T],
        *,
        max_tokens: int = 16000,
    ) -> T:
        """Return a parsed pydantic instance of ``output_type``.

        Augments the user message with the JSON schema and the literal word
        "JSON". Providers that honor OpenAI's ``json_schema`` mode ignore
        this — they already get the schema via ``response_format``. Providers
        that downgrade to ``json_object`` mode (notably Qwen/Alibaba via
        OpenRouter) need the schema in-prompt or they don't know what fields
        to emit, and refuse outright unless the word "JSON" appears.
        """
        schema_text = json.dumps(output_type.model_json_schema(), indent=2)
        augmented_user = (
            f"{user}\n\n"
            f"Respond with a single JSON object matching this schema:\n"
            f"```json\n{schema_text}\n```"
        )
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": augmented_user},
            ],
            response_format=output_type,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"LLM did not produce a parseable {output_type.__name__}; "
                f"finish_reason={response.choices[0].finish_reason}"
            )
        return parsed
