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
MAX_FAILURE_EXAMPLES = 3


@dataclass
class FailureExample:
    """One concrete failed benchmark case, stored as repr strings for safety."""

    input_repr: str
    expected_repr: str
    actual_repr: str | None = None
    error: str | None = None


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
    failure_examples: list[FailureExample] = field(default_factory=list)


@dataclass
class GraderOutcome:
    """Subprocess grader output with backwards-compatible tuple unpacking."""

    results: list[bool]
    error: str | None = None
    failure_examples: list[FailureExample] = field(default_factory=list)

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.results
        yield self.error


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
    atol = float(problem.get("atol") or 0.0)

    plus_outcome = _normalize_grader_outcome(
        _run_grader(
            code,
            entry_point,
            plus_inputs,
            plus_expected,
            atol=atol,
        )
    )
    base_outcome = _normalize_grader_outcome(
        _run_grader(
            code,
            entry_point,
            base_inputs,
            base_expected,
            atol=atol,
        )
    )
    plus_results, plus_err = plus_outcome.results, plus_outcome.error
    base_results, base_err = base_outcome.results, base_outcome.error

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
        failure_examples=plus_outcome.failure_examples,
    )


def _normalize_grader_outcome(
    outcome: GraderOutcome | tuple[list[bool], str | None],
) -> GraderOutcome:
    if isinstance(outcome, GraderOutcome):
        return outcome
    results, error = outcome
    return GraderOutcome(results=list(results), error=error)


# Harness template: user code at module top level, then iterate inputs
# and print pass/fail booleans behind a marker so we can find them
# even if the user code itself prints to stdout.
_HARNESS = '''\
import math
import sys

# ----- USER CODE -----
{code}
# ----- END USER CODE -----

# ``repr(float("inf"))`` is ``inf``. Define these names so EvalPlus cases
# containing infinities remain valid Python literals inside this harness.
inf = float("inf")
nan = float("nan")

_inputs = {inputs}
_expected = {expected}
_atol = {atol}
_failure_limit = {failure_limit}

def _idcs_equal(actual, expected, atol):
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _idcs_equal(a, e, atol) for a, e in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _idcs_equal(actual[key], expected[key], atol) for key in expected
        )
    return actual == expected

def _idcs_short_repr(value):
    text = repr(value)
    return text if len(text) <= 300 else text[:297] + "..."

_results = []
_failures = []
for _inp, _exp in zip(_inputs, _expected):
    try:
        _actual = {entry}(*_inp)
        _ok = _idcs_equal(_actual, _exp, _atol)
        _results.append(_ok)
        if not _ok and len(_failures) < _failure_limit:
            _failures.append({{
                "input_repr": _idcs_short_repr(_inp),
                "expected_repr": _idcs_short_repr(_exp),
                "actual_repr": _idcs_short_repr(_actual),
            }})
    except BaseException as _exc:
        _results.append(False)
        if len(_failures) < _failure_limit:
            _failures.append({{
                "input_repr": _idcs_short_repr(_inp),
                "expected_repr": _idcs_short_repr(_exp),
                "error": type(_exc).__name__ + ": " + str(_exc)[:200],
            }})

import json
sys.stdout.write({marker!r} + json.dumps({{"results": _results, "failures": _failures}}))
'''


def _run_grader(
    code: str,
    entry: str,
    inputs: list[Any],
    expected: list[Any],
    atol: float = 0.0,
    timeout_s: float = SUBPROCESS_TIMEOUT,
    failure_example_limit: int = MAX_FAILURE_EXAMPLES,
) -> GraderOutcome:
    if not inputs:
        return GraderOutcome([], None)

    try:
        harness = _HARNESS.format(
            code=code,
            inputs=repr(inputs),
            expected=repr(expected),
            atol=repr(atol),
            entry=entry,
            marker=SCORE_MARKER,
            failure_limit=repr(failure_example_limit),
        )
    except Exception as e:
        return GraderOutcome([False] * len(inputs), f"harness format failed: {e}")

    with tempfile.TemporaryDirectory(prefix="idcs-grader-") as tmpdir:
        script_path = Path(tmpdir) / "harness.py"
        script_path.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return GraderOutcome([False] * len(inputs), "subprocess timeout")

    idx = proc.stdout.rfind(SCORE_MARKER)
    if idx == -1:
        tail = (proc.stderr.strip().splitlines() or [""])[-1]
        return GraderOutcome(
            [False] * len(inputs),
            f"no scores in stdout (stderr: {tail[:200]})",
        )
    try:
        payload = json.loads(proc.stdout[idx + len(SCORE_MARKER) :])
    except json.JSONDecodeError as e:
        return GraderOutcome([False] * len(inputs), f"JSON parse: {e}")
    if isinstance(payload, list):
        return GraderOutcome(results=[bool(r) for r in payload])
    if not isinstance(payload, dict):
        return GraderOutcome(
            [False] * len(inputs),
            f"unexpected payload type: {type(payload).__name__}",
        )
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return GraderOutcome([False] * len(inputs), "missing results list")
    examples = [
        FailureExample(
            input_repr=str(item.get("input_repr", "")),
            expected_repr=str(item.get("expected_repr", "")),
            actual_repr=(
                str(item["actual_repr"]) if item.get("actual_repr") is not None else None
            ),
            error=str(item["error"]) if item.get("error") is not None else None,
        )
        for item in payload.get("failures", [])
        if isinstance(item, dict)
    ]
    return GraderOutcome(
        results=[bool(r) for r in raw_results],
        failure_examples=examples,
    )


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
    failure_examples: list[FailureExample] = []
    for test in task.tests:
        try:
            exec(test.code, namespace)  # noqa: S102
            passed += 1
        except AssertionError as e:
            errors.append(f"{test.id}: AssertionError: {e}")
            if len(failure_examples) < MAX_FAILURE_EXAMPLES:
                failure_examples.append(
                    FailureExample(
                        input_repr=test.id,
                        expected_repr="assertion passes",
                        error=f"AssertionError: {e}",
                    )
                )
        except Exception as e:
            errors.append(f"{test.id}: {type(e).__name__}: {e}")
            if len(failure_examples) < MAX_FAILURE_EXAMPLES:
                failure_examples.append(
                    FailureExample(
                        input_repr=test.id,
                        expected_repr="test executes",
                        error=f"{type(e).__name__}: {e}",
                    )
                )

    total = len(task.tests)
    return ScoreResult(
        pass_count=passed,
        total_count=total,
        pass_rate=passed / total,
        errors=errors,
        failure_examples=failure_examples,
    )
