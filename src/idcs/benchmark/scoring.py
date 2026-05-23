"""Score generated code against a task's tests.

Two paths, dispatched on ``task.id``:

- **MBPP+ tasks** (``Mbpp/...``) — we use EvalPlus for the test **data**
  (``plus_input`` cases plus the expected outputs from
  ``get_groundtruth``) but run user code in *our own subprocess*. EvalPlus's
  in-process grader uses ``reliability_guard`` which calls ``setrlimit``
  for ``RLIMIT_AS``; macOS doesn't honor that for non-root processes, so
  the worker dies before producing results. Subprocess isolation is
  weaker than EvalPlus's grader but works on every platform, is
  crash-safe, and the scoring logic is trivial — comparing actual to
  ``get_groundtruth``'s expected output per input.
- **Seed-corpus tasks** (``seed/...``) — local ``exec`` in this process
  (trusted short asserts we wrote).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from idcs.schemas import Task

SUBPROCESS_TIMEOUT = 30.0
SCORE_MARKER = "__IDCS_SCORES__"


@dataclass
class ScoreResult:
    """Per-task scoring outcome.

    ``pass_rate`` is the headline number — for MBPP+ tasks it's the
    ``plus_input`` pass fraction. ``base_pass_rate`` is reported for
    diagnostics: base=1.0 / plus=0.4 confirms a pattern-match.
    """

    pass_count: int
    total_count: int
    pass_rate: float
    base_pass_rate: float | None = None
    errors: list[str] = field(default_factory=list)


def score(task: Task, code: str) -> float:
    return score_detailed(task, code).pass_rate


def score_detailed(task: Task, code: str) -> ScoreResult:
    if task.id.startswith("Mbpp/"):
        return _score_mbpp_plus(task, code)
    return _score_local(task, code)


# ---- MBPP+: EvalPlus data, our subprocess ----


@lru_cache(maxsize=1)
def _mbpp_groundtruth() -> tuple[dict[str, Any], dict[str, Any]]:
    """Memoized (problems, expected_output) for MBPP+.

    ``get_groundtruth`` runs the canonical solutions to derive expected
    outputs; cache it for the process lifetime.
    """
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.evaluate import MBPP_OUTPUT_NOT_NONE_TASKS, get_groundtruth

    problems = get_mbpp_plus()
    expected = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
    return problems, expected


def _score_mbpp_plus(task: Task, code: str) -> ScoreResult:
    problems, expected = _mbpp_groundtruth()
    if task.id not in problems:
        raise ValueError(f"{task.id!r} not in MBPP+; cannot grade.")

    problem = problems[task.id]
    entry_point = problem.get("entry_point") or ""
    if not entry_point:
        return ScoreResult(0, 0, 0.0, errors=["MBPP+ task missing entry_point"])

    plus_inputs = list(problem.get("plus_input") or [])
    plus_expected = list(expected[task.id].get("plus") or [])
    base_inputs = list(problem.get("base_input") or [])
    base_expected = list(expected[task.id].get("base") or [])

    plus_results, plus_err = _run_grader(code, entry_point, plus_inputs, plus_expected)
    base_results, base_err = _run_grader(code, entry_point, base_inputs, base_expected)

    plus_passed = sum(1 for r in plus_results if r)
    plus_total = len(plus_results)
    base_passed = sum(1 for r in base_results if r)
    base_total = len(base_results)

    errors: list[str] = []
    if plus_err:
        errors.append(f"plus: {plus_err}")
    # Don't double-report — if user code didn't even define entry_point,
    # both runs will hit the same error.
    elif base_err:
        errors.append(f"base: {base_err}")

    return ScoreResult(
        pass_count=plus_passed,
        total_count=plus_total,
        pass_rate=(plus_passed / plus_total) if plus_total > 0 else 0.0,
        base_pass_rate=(base_passed / base_total) if base_total > 0 else None,
        errors=errors,
    )


# Harness template: user code at module top level, then iterate inputs
# and print pass/fail booleans behind a marker so we can find them
# even if the user code itself prints to stdout.
_HARNESS = '''\
import sys

# ----- USER CODE -----
{code}
# ----- END USER CODE -----

_inputs = {inputs}
_expected = {expected}
_results = []
for _inp, _exp in zip(_inputs, _expected):
    try:
        _actual = {entry}(*_inp)
        _results.append(_actual == _exp)
    except BaseException:
        _results.append(False)

import json
sys.stdout.write({marker!r} + json.dumps(_results))
'''


def _run_grader(
    code: str,
    entry: str,
    inputs: list[Any],
    expected: list[Any],
    timeout_s: float = SUBPROCESS_TIMEOUT,
) -> tuple[list[bool], str | None]:
    if not inputs:
        return [], None

    try:
        harness = _HARNESS.format(
            code=code,
            inputs=repr(inputs),
            expected=repr(expected),
            entry=entry,
            marker=SCORE_MARKER,
        )
    except Exception as e:
        return [False] * len(inputs), f"harness format failed: {e}"

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
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
        except subprocess.TimeoutExpired:
            return [False] * len(inputs), "subprocess timeout"
    finally:
        script_path.unlink(missing_ok=True)

    idx = proc.stdout.rfind(SCORE_MARKER)
    if idx == -1:
        tail = (proc.stderr.strip().splitlines() or [""])[-1]
        return [False] * len(inputs), f"no scores in stdout (stderr: {tail[:200]})"
    try:
        results = json.loads(proc.stdout[idx + len(SCORE_MARKER) :])
    except json.JSONDecodeError as e:
        return [False] * len(inputs), f"JSON parse: {e}"
    if not isinstance(results, list):
        return [False] * len(inputs), f"unexpected payload type: {type(results).__name__}"
    return [bool(r) for r in results], None


# ---- Seed corpus: local exec for trusted short asserts ----


def _score_local(task: Task, code: str) -> ScoreResult:
    if not task.tests:
        return ScoreResult(pass_count=0, total_count=0, pass_rate=1.0)

    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        return ScoreResult(
            pass_count=0,
            total_count=len(task.tests),
            pass_rate=0.0,
            errors=[f"Code execution failed: {type(e).__name__}: {e}"],
        )

    passed = 0
    errors: list[str] = []
    for test in task.tests:
        try:
            exec(test.code, namespace)  # noqa: S102
            passed += 1
        except AssertionError as e:
            errors.append(f"{test.id}: AssertionError: {e}")
        except Exception as e:
            errors.append(f"{test.id}: {type(e).__name__}: {e}")

    total = len(task.tests)
    return ScoreResult(
        pass_count=passed,
        total_count=total,
        pass_rate=passed / total,
        errors=errors,
    )

