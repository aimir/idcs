"""Sandboxed test execution for benchmark scoring.

Phase 0 stub: subprocess + wall-clock timeout, no resource limits, no
filesystem isolation. Adequate for code we generate ourselves and for
trusted benchmark suites (MBPP+, HumanEval+). Replace with firejail,
bubblewrap, or a container runtime before running anything untrusted.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    """Outcome of running one piece of code in the sandbox."""

    passed: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool


def run_python(code: str, *, timeout_s: float = 5.0) -> RunResult:
    """Execute `code` as a Python script and return the result."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = Path(f.name)
    try:
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(
                passed=False,
                stdout=e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
                stderr=e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
                return_code=-1,
                timed_out=True,
            )
        return RunResult(
            passed=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
            timed_out=False,
        )
    finally:
        script_path.unlink(missing_ok=True)
