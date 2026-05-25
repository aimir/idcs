"""Tests for the local Codex CLI LLM backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from idcs.llm import DEFAULT_CODEX_MODEL, LLM, BudgetExceededError, runtime_snapshot


class Answer(BaseModel):
    value: int


def test_codex_backend_runs_codex_exec_without_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("codex completion", encoding="utf-8")
        calls.append(
            {
                "command": command,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="status noise", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.setenv("IDCS_CODEX_MODEL", "test-codex-model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    client = LLM()
    result = client.complete("system prompt", "user prompt")

    assert result == "codex completion"
    assert client.calls_made == 1
    assert calls[0]["command"][:4] == ["codex", "exec", "--model", "test-codex-model"]
    assert "--ephemeral" in calls[0]["command"]
    assert "--ignore-user-config" in calls[0]["command"]
    assert "--output-last-message" in calls[0]["command"]
    assert calls[0]["command"][-1] == "-"
    assert "stateless completion backend" in calls[0]["input"]
    assert "SYSTEM:\nsystem prompt" in calls[0]["input"]
    assert "USER:\nuser prompt" in calls[0]["input"]


def test_codex_backend_defaults_to_small_model(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("ok", encoding="utf-8")
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("IDCS_CODEX_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    assert LLM().complete("system", "user") == "ok"
    assert commands[0][commands[0].index("--model") + 1] == DEFAULT_CODEX_MODEL


def test_codex_backend_accepts_fast_service_tier_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("ok", encoding="utf-8")
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.setenv("IDCS_CODEX_SERVICE_TIER", "fast")
    monkeypatch.setenv("IDCS_CODEX_REASONING_EFFORT", "none")
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    assert LLM().complete("system", "user") == "ok"
    assert "-c" in commands[0]
    assert 'service_tier="fast"' in commands[0]
    assert 'model_reasoning_effort="none"' in commands[0]


def test_codex_complete_typed_parses_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("```json\n{\"value\": 7}\n```", encoding="utf-8")
        prompts.append(input)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    llm = LLM()
    result = llm.complete_typed("system", "give me json", Answer)

    assert result == Answer(value=7)
    assert "Respond with a single JSON object" in prompts[0]
    assert '"value"' in prompts[0]
    # codex has no structured-output API, so every typed call is recorded
    # as a fallback for telemetry parity with the openai-compatible path.
    assert llm.structured_fallback_count == 1


def test_codex_complete_typed_repairs_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    outputs = ['{"value": "broken}', '{"value": 7}']

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(outputs.pop(0), encoding="utf-8")
        prompts.append(input)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    llm = LLM()
    result = llm.complete_typed("system", "give me json", Answer)

    assert result == Answer(value=7)
    assert llm.calls_made == 2
    assert llm.structured_fallback_count == 1
    assert "could not be parsed" in prompts[1]
    assert "Return only one valid JSON object" in prompts[1]


def test_codex_complete_typed_retries_json_repair_once_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    outputs = ['{"value": "broken}', '{"value": "still broken}', '{"value": 7}']

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(outputs.pop(0), encoding="utf-8")
        prompts.append(input)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    llm = LLM()
    result = llm.complete_typed("system", "give me json", Answer)

    assert result == Answer(value=7)
    assert llm.calls_made == 3
    assert llm.structured_fallback_count == 2
    assert len(prompts) == 3


def test_codex_failure_omits_captured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-test-secret-value"

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, timeout, check
        return subprocess.CompletedProcess(
            command,
            2,
            stdout=f"stdout contains {secret}",
            stderr=f"stderr contains {secret}",
        )

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)
    monkeypatch.setattr("idcs.llm.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError) as exc_info:
        LLM().complete("system", "user")

    message = str(exc_info.value)
    assert secret not in message
    assert "output omitted" in message


def test_codex_missing_binary_does_not_consume_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("codex not installed")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    llm = LLM()
    with pytest.raises(RuntimeError, match="codex.*not found"):
        llm.complete("system", "user")
    # Binary missing: no API call happened, budget must be untouched.
    assert llm.calls_made == 0


def test_codex_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, timeout, check
        attempts.append(1)
        if len(attempts) < 3:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="transient")
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("recovered", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)
    monkeypatch.setattr("idcs.llm.time.sleep", lambda _s: None)

    llm = LLM()
    assert llm.complete("system", "user") == "recovered"
    assert len(attempts) == 3
    # Two failed attempts + one successful attempt all count against budget.
    assert llm.calls_made == 3


def test_codex_backend_respects_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, timeout, check
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.setattr("idcs.llm.subprocess.run", fake_run)

    llm = LLM(max_calls=1)
    assert llm.complete("system", "user") == "ok"
    with pytest.raises(BudgetExceededError):
        llm.complete("system", "user")
    assert llm.calls_made == 1


def test_runtime_snapshot_records_codex_model_and_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IDCS_BACKEND", "codex")
    monkeypatch.setenv("IDCS_CODEX_MODEL", "gpt-test")
    monkeypatch.setenv("IDCS_CODEX_TIMEOUT_S", "123")
    monkeypatch.setenv("IDCS_CODEX_SERVICE_TIER", "fast")
    monkeypatch.setenv("IDCS_CODEX_REASONING_EFFORT", "none")

    assert runtime_snapshot() == {
        "backend": "codex",
        "model": "gpt-test",
        "codex_timeout_s": 123.0,
        "codex_service_tier": "fast",
        "codex_reasoning_effort": "none",
    }


def test_runtime_snapshot_records_openrouter_model_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IDCS_BACKEND", raising=False)
    monkeypatch.setenv("IDCS_MODEL", "provider/model")
    monkeypatch.setenv("IDCS_API_KEY", "secret-value")

    snapshot = runtime_snapshot()

    assert snapshot["backend"] == "openrouter"
    assert snapshot["model"] == "provider/model"
    assert snapshot["base_url"] == "https://openrouter.ai/api/v1"
    assert snapshot["require_parameters"] is True
    assert "secret-value" not in str(snapshot)
