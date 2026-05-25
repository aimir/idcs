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
import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Load .env so OPENROUTER_API_KEY / IDCS_MODEL are available to LLM().
# Safe to call here — openai's SDK reads env vars at client instantiation,
# not at import time.
load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5


class BudgetExceededError(RuntimeError):
    """Raised when an ``LLM`` instance has hit its ``max_calls`` ceiling."""


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
        max_calls: int | None = None,
    ) -> None:
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )
        self.model = model or os.environ.get("IDCS_MODEL") or DEFAULT_MODEL
        self.require_parameters = require_parameters
        # Budget tracking. Every API call attempt (success or failure)
        # increments ``calls_made``. Once it would exceed ``max_calls`` the
        # next call raises ``BudgetExceededError`` instead of contacting
        # the provider. ``None`` means no cap.
        self.max_calls = max_calls
        self.calls_made = 0
        self.structured_fallback_count = 0

    def _check_budget(self) -> None:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise BudgetExceededError(
                f"LLM call budget exhausted ({self.calls_made}/{self.max_calls})"
            )

    def _with_retry(self, fn: Callable[[], T_ret]) -> T_ret:
        """Call ``fn()`` with budget tracking + exponential backoff on 429/5xx.

        Counts every attempt against ``calls_made``: a 429 retry costs as
        much money as a successful call, so it should count too.
        """
        for attempt in range(MAX_RETRIES + 1):
            self._check_budget()
            try:
                result = fn()
                self.calls_made += 1
                return result
            except openai.RateLimitError:
                self.calls_made += 1
                if attempt == MAX_RETRIES:
                    raise
                delay = (2**attempt) + random.uniform(0, 1)
                log.warning(
                    "429 rate-limited, retry %d/%d in %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
            except openai.APIStatusError as e:
                self.calls_made += 1
                if e.status_code < 500 or attempt == MAX_RETRIES:
                    raise
                delay = (2**attempt) + random.uniform(0, 1)
                log.warning(
                    "%d server error, retry %d/%d in %.1fs",
                    e.status_code, attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

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
        response = self._with_retry(
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

        Tries ``beta.chat.completions.parse`` first (works with Claude, GPT-4o).
        Falls back to plain completion + manual JSON extraction for providers
        like Qwen that echo back the schema instead of conforming to it.
        """
        schema_text = json.dumps(output_type.model_json_schema(), indent=2)
        augmented_user = (
            f"{user}\n\n"
            f"Respond with a single JSON object that **conforms to** the schema "
            f"below. You must produce an INSTANCE — concrete values for each "
            f"field — not the schema definition itself. Do not include `$schema`, "
            f"`type: object`, `properties`, `required`, or `$defs` keys in your "
            f"response.\n\n"
            f"Schema:\n```json\n{schema_text}\n```"
        )
        fallback_reason: str | None = None
        calls_before_parse = self.calls_made
        try:
            response = self._with_retry(
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
            if parsed is not None:
                return parsed
            fallback_reason = "parsed_none"
        except (ValidationError, KeyError):
            if self.calls_made == calls_before_parse:
                self.calls_made += 1
            fallback_reason = "parse_exception"

        self._record_structured_fallback(output_type.__name__, fallback_reason or "unknown")

        raw = self.complete(system, augmented_user, max_tokens=max_tokens)
        return _parse_json_response(raw, output_type)

    def _record_structured_fallback(self, output_type_name: str, reason: str) -> None:
        self.structured_fallback_count += 1
        log.warning(
            "Structured parse fallback for %s (%s); falling back to text extraction",
            output_type_name,
            reason,
        )


def _parse_json_response(raw: str, output_type: type[T]) -> T:
    """Extract JSON from raw LLM text (strips markdown fences) and validate."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Handle case where model wraps JSON in other text — find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse model output as JSON ({e}). "
            f"Raw response (first 500 chars):\n{raw[:500]}"
        ) from e

    # Some providers (notably Qwen via Alibaba) ignore the "produce an
    # instance" instruction and echo the schema we sent them. Catch that
    # explicitly so the error is actionable rather than a confusing pydantic
    # "field required" message about the schema metadata.
    if isinstance(data, dict) and _looks_like_schema_echo(data):
        raise RuntimeError(
            f"Model returned a JSON schema instead of a {output_type.__name__} "
            f"instance. This is a known failure mode of some providers — "
            f"set IDCS_MODEL to a model with reliable structured-output "
            f"support (e.g. anthropic/claude-sonnet-4.5, openai/gpt-4o, "
            f"google/gemini-2.5-pro) and retry."
        )

    return output_type.model_validate(data)


def _looks_like_schema_echo(data: dict[str, Any]) -> bool:
    """Heuristic: did the provider echo back the JSON schema verbatim?"""
    if "$schema" in data:
        return True
    # Bare schema shape: ``type: object`` + ``properties`` at top level
    return data.get("type") == "object" and "properties" in data
