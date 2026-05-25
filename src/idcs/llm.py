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
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
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
DEFAULT_BACKEND = "openrouter"
CODEX_BACKEND = "codex"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CODEX_TIMEOUT_S = 300.0
MAX_RETRIES = 5
JSON_REPAIR_ATTEMPTS = 2


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
        backend: str | None = None,
    ) -> None:
        self.backend = (backend or os.environ.get("IDCS_BACKEND") or DEFAULT_BACKEND).lower()
        if self.backend == CODEX_BACKEND:
            self.client: openai.OpenAI | None = None
            self.model = model or os.environ.get("IDCS_CODEX_MODEL") or DEFAULT_CODEX_MODEL
            self.codex_timeout_s = _float_env("IDCS_CODEX_TIMEOUT_S", DEFAULT_CODEX_TIMEOUT_S)
        elif self.backend in {"openrouter", "openai", "openai-compatible"}:
            resolved_base_url = os.environ.get("IDCS_BASE_URL") or base_url
            resolved_api_key = api_key or os.environ.get("IDCS_API_KEY")
            if resolved_api_key is None:
                resolved_api_key = os.environ.get("OPENROUTER_API_KEY")
            if resolved_api_key is None and self.backend == "openai":
                resolved_api_key = os.environ.get("OPENAI_API_KEY")
            self.client = openai.OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
            )
            self.model = model or os.environ.get("IDCS_MODEL") or DEFAULT_MODEL
            self.codex_timeout_s = DEFAULT_CODEX_TIMEOUT_S
        else:
            raise ValueError(
                f"Unsupported IDCS_BACKEND={self.backend!r}; expected 'openrouter' or 'codex'."
            )
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
        if self.backend == CODEX_BACKEND:
            return self._complete_with_codex(system, user, max_tokens=max_tokens)

        if self.client is None:
            raise RuntimeError("OpenAI-compatible backend was not initialized.")
        client = self.client
        response = self._with_retry(
            lambda: client.chat.completions.create(
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
        if self.backend == CODEX_BACKEND:
            # Codex has no structured-output API; we always parse JSON from
            # text. Record it so telemetry reflects that every codex typed
            # call is on the fallback path.
            self._record_structured_fallback(output_type.__name__, "codex_backend")
            raw = self.complete(system, augmented_user, max_tokens=max_tokens)
            return _parse_json_response(raw, output_type)

        fallback_reason: str | None = None
        calls_before_parse = self.calls_made
        try:
            if self.backend == CODEX_BACKEND:
                raw = self.complete(system, augmented_user, max_tokens=max_tokens)
                return self._parse_typed_with_repair(
                    system,
                    augmented_user,
                    raw,
                    output_type,
                    max_tokens=max_tokens,
                )

            if self.client is None:
                raise RuntimeError("OpenAI-compatible backend was not initialized.")
            client = self.client
            response = self._with_retry(
                lambda: client.beta.chat.completions.parse(
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
        return self._parse_typed_with_repair(
            system,
            augmented_user,
            raw,
            output_type,
            max_tokens=max_tokens,
        )

    def _parse_typed_with_repair(
        self,
        system: str,
        original_user: str,
        raw: str,
        output_type: type[T],
        *,
        max_tokens: int,
    ) -> T:
        """Parse typed JSON, retrying repair prompts when a text backend emits invalid JSON."""
        try:
            return _parse_json_response(raw, output_type)
        except RuntimeError as exc:
            last_raw = raw
            last_error = str(exc)

        for _ in range(JSON_REPAIR_ATTEMPTS):
            self._record_structured_fallback(output_type.__name__, "json_repair")
            repair_user = _json_repair_prompt(
                original_user,
                last_raw,
                last_error,
                output_type,
            )
            repaired = self.complete(system, repair_user, max_tokens=max_tokens)
            try:
                return _parse_json_response(repaired, output_type)
            except RuntimeError as exc:
                last_raw = repaired
                last_error = str(exc)

        raise RuntimeError(last_error)

    def _record_structured_fallback(self, output_type_name: str, reason: str) -> None:
        self.structured_fallback_count += 1
        log.warning(
            "Structured parse fallback for %s (%s); falling back to text extraction",
            output_type_name,
            reason,
        )

    def _complete_with_codex(self, system: str, user: str, *, max_tokens: int) -> str:
        # max_tokens is advisory only: the codex CLI does not expose a hard
        # cap, so it gets passed to the model as prose. Long-running prompts
        # can exceed the budget — callers that need a strict ceiling should
        # use the openai-compatible backend.
        prompt = (
            "You are serving as a stateless completion backend for a benchmark runner.\n"
            "Do not edit files, run shell commands, or inspect the repository.\n"
            "Return only the requested answer.\n\n"
            f"SYSTEM:\n{system}\n\n"
            f"USER:\n{user}\n\n"
            f"Keep the response within roughly {max_tokens} tokens."
        )
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            self._check_budget()
            try:
                result = self._codex_subprocess_once(prompt)
                self.calls_made += 1
                return result
            except FileNotFoundError as exc:
                # Binary missing is not a transient failure and no API call
                # was made — do not retry and do not bill the budget.
                raise RuntimeError(
                    "Codex backend requested but the `codex` executable was not found."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                self.calls_made += 1
                last_error = RuntimeError(
                    f"Codex backend timed out after {self.codex_timeout_s:g} seconds."
                )
                last_error.__cause__ = exc
            except RuntimeError as exc:
                self.calls_made += 1
                last_error = exc
            if attempt == MAX_RETRIES:
                assert last_error is not None
                raise last_error
            delay = (2**attempt) + random.uniform(0, 1)
            log.warning(
                "Codex backend error, retry %d/%d in %.1fs",
                attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _codex_subprocess_once(self, prompt: str) -> str:
        with TemporaryDirectory(prefix="idcs-codex-") as temp_dir:
            output_path = Path(temp_dir) / "last-message.txt"
            # ``-C`` puts the codex process's working directory at the
            # current repo root. With ``--sandbox read-only`` the process
            # cannot write, but it can still read repo contents (including
            # any ``.env`` / secrets) and could exfiltrate via its own API
            # calls. Acceptable for a benchmark backend run on a trusted
            # checkout; revisit if codex grows reach beyond CWD.
            command = [
                os.environ.get("IDCS_CODEX_EXECUTABLE", "codex"),
                "exec",
                "--model",
                self.model,
                *_codex_config_args(),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--color",
                "never",
                "-C",
                str(Path.cwd()),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.codex_timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Codex backend failed with exit code "
                    f"{completed.returncode}; output omitted to avoid leaking prompts "
                    "or credentials."
                )
            if output_path.exists():
                output = output_path.read_text(encoding="utf-8").strip()
                if output:
                    return output
            raise RuntimeError("Codex backend returned an empty final message.")


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def runtime_snapshot(llm: Any | None = None) -> dict[str, Any]:
    """Return non-secret LLM runtime metadata for benchmark artifacts."""
    backend = str(
        getattr(llm, "backend", None)
        or os.environ.get("IDCS_BACKEND")
        or DEFAULT_BACKEND
    ).lower()
    model = getattr(llm, "model", None)
    if model is None:
        model = (
            os.environ.get("IDCS_CODEX_MODEL")
            if backend == CODEX_BACKEND
            else os.environ.get("IDCS_MODEL")
        )
    if model is None:
        model = DEFAULT_CODEX_MODEL if backend == CODEX_BACKEND else DEFAULT_MODEL

    snapshot: dict[str, Any] = {
        "backend": backend,
        "model": model,
    }
    if backend == CODEX_BACKEND:
        snapshot.update(
            {
                "codex_timeout_s": getattr(
                    llm,
                    "codex_timeout_s",
                    _float_env("IDCS_CODEX_TIMEOUT_S", DEFAULT_CODEX_TIMEOUT_S),
                ),
                "codex_service_tier": os.environ.get("IDCS_CODEX_SERVICE_TIER"),
                "codex_reasoning_effort": os.environ.get(
                    "IDCS_CODEX_REASONING_EFFORT"
                ),
            }
        )
    else:
        snapshot.update(
            {
                "base_url": os.environ.get("IDCS_BASE_URL") or DEFAULT_BASE_URL,
                "require_parameters": getattr(llm, "require_parameters", True),
            }
        )
    return snapshot


def _codex_config_args() -> list[str]:
    """Render optional Codex CLI config overrides for local benchmark calls."""
    args: list[str] = []
    config_env = {
        "service_tier": os.environ.get("IDCS_CODEX_SERVICE_TIER"),
        "model_reasoning_effort": os.environ.get("IDCS_CODEX_REASONING_EFFORT"),
    }
    for key, value in config_env.items():
        if value:
            args.extend(["-c", f"{key}={json.dumps(value)}"])
    return args


def _json_repair_prompt(
    original_user: str,
    raw: str,
    error: str,
    output_type: type[BaseModel],
) -> str:
    return (
        "Your previous response could not be parsed as the requested JSON object.\n"
        f"Output type: {output_type.__name__}\n"
        f"Parse error:\n{error[:1000]}\n\n"
        "Original request:\n"
        f"{original_user}\n\n"
        "Previous response:\n"
        "```text\n"
        f"{raw[:4000]}\n"
        "```\n\n"
        "Return only one valid JSON object. Do not include markdown fences or commentary."
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
