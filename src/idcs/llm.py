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
import random
import time
from typing import Any, Callable, Protocol, TypeVar

from dotenv import load_dotenv

load_dotenv()

import openai
from pydantic import BaseModel

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5


def _with_retry(fn: Callable[[], T_ret], max_retries: int = MAX_RETRIES) -> T_ret:
    """Call fn(), retrying on transient 429/5xx with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except openai.RateLimitError:
            if attempt == max_retries:
                raise
        except openai.APIStatusError as e:
            if e.status_code < 500 or attempt == max_retries:
                raise
        delay = (2**attempt) + random.uniform(0, 1)
        time.sleep(delay)
    raise AssertionError("unreachable")

T = TypeVar("T", bound=BaseModel)
T_ret = TypeVar("T_ret")


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
        require_parameters: bool = True,
    ) -> None:
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )
        self.model = model or os.environ.get("IDCS_MODEL") or DEFAULT_MODEL
        self.require_parameters = require_parameters

    @property
    def _extra_body(self) -> dict[str, Any]:
        """Provider-routing hints for OpenRouter.

        ``require_parameters: true`` forces OpenRouter to only route to a
        provider that supports every parameter we send. Critical for
        ``response_format: json_schema`` — many models are multi-provider
        on OpenRouter and at least one provider per model often downgrades
        to ``json_object`` mode silently.
        """
        if not self.require_parameters:
            return {}
        return {"provider": {"require_parameters": True}}

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 16000,
    ) -> str:
        """Return the assistant message's text content (empty string if absent)."""
        response = _with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=self._extra_body,
            )
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

        Belt and suspenders: ``response_format`` carries the schema to
        providers that honor ``json_schema`` mode, and a copy of the schema
        plus the literal word "JSON" goes into the user message for providers
        that downgrade to ``json_object`` mode. OpenRouter's
        ``require_parameters`` filter checks parameter *names* (does the
        provider accept ``response_format``?) not subtypes (does it honor
        ``json_schema``?) — Alibaba/Qwen passes the filter and then strips
        the schema, so the in-prompt copy is the actual safety net.
        """
        schema_text = json.dumps(output_type.model_json_schema(), indent=2)
        augmented_user = (
            f"{user}\n\n"
            f"Respond with a single JSON object matching this schema:\n"
            f"```json\n{schema_text}\n```"
        )
        response = _with_retry(
            lambda: self.client.beta.chat.completions.parse(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": augmented_user},
                ],
                response_format=output_type,
                extra_body=self._extra_body,
            )
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"LLM did not produce a parseable {output_type.__name__}; "
                f"finish_reason={response.choices[0].finish_reason}"
            )
        return parsed
