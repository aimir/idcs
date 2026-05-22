"""Smoke tests for the sandbox stub."""

from __future__ import annotations

from idcs.benchmark.runner import run_python


def test_run_python_success() -> None:
    result = run_python("print('hello')")
    assert result.passed
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_python_assertion_failure() -> None:
    result = run_python("assert 1 == 2")
    assert not result.passed
    assert result.return_code != 0
    assert not result.timed_out


def test_run_python_timeout() -> None:
    result = run_python("import time; time.sleep(10)", timeout_s=0.3)
    assert result.timed_out
    assert not result.passed
