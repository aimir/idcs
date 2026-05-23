"""Score generated code against a task's tests.

Two paths:

- **MBPP+ tasks** (``Mbpp/...`` ids) → EvalPlus's ``check_correctness``,
  scoring against the ``plus_input`` set only (the harder cases designed to
  catch implementations that pattern-match prompt examples without
  understanding the task). EvalPlus runs the code in a subprocess; we
  don't host code execution here.
- **Seed-corpus tasks** (``seed/...`` ids) → local ``exec``. Trusted: tests
  are short asserts we wrote ourselves; code is short and constrained to
  the seed task. No subprocess overhead; fine for the size of the corpus.

``score(task, code)`` dispatches on ``task.id`` so callers don't care.

**macOS workaround**: EvalPlus's ``reliability_guard`` tries to set
``RLIMIT_AS`` which non-root processes cannot do on macOS, killing the
worker. We force ``multiprocessing`` to ``fork`` (so the in-parent stub
propagates to the worker) and stub the memory limit out. Removable once
upstream EvalPlus handles macOS.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from idcs.schemas import Task


@lru_cache(maxsize=1)
def _apply_macos_compat() -> None:
    """No-op off macOS; otherwise patch EvalPlus to survive RLIMIT_AS."""
    if sys.platform != "darwin":
        return
    with contextlib.suppress(RuntimeError, ValueError):
        multiprocessing.set_start_method("fork", force=True)
    try:
        from evalplus.eval import utils as _eu

        _original = _eu.reliability_guard

        def _macos_safe(maximum_memory_bytes: int | None = None) -> None:
            # Skip the RLIMIT_AS / RLIMIT_DATA path; keep upstream's
            # builtin-disabling effects by calling with None.
            _original(maximum_memory_bytes=None)

        _eu.reliability_guard = _macos_safe
    except ImportError:
        pass


@dataclass
class ScoreResult:
    """Per-task scoring outcome.

    ``pass_rate`` is the headline number. For MBPP+ it's the fraction of
    ``plus_input`` cases that passed. ``base_pass_rate`` is reported for
    diagnostics — seeing base=1.0, plus=0.4 confirms a pattern-match.
    """

    pass_count: int
    total_count: int
    pass_rate: float
    base_pass_rate: float | None = None
    errors: list[str] = field(default_factory=list)


def score(task: Task, code: str) -> float:
    """Headline pass rate in [0, 1]. Dispatches by task source."""
    return score_detailed(task, code).pass_rate


def score_detailed(task: Task, code: str) -> ScoreResult:
    """Detailed score. Dispatches by task source."""
    if task.id.startswith("Mbpp/"):
        return _score_mbpp_plus(task, code)
    return _score_local(task, code)


# ---- MBPP+ via EvalPlus ----


@lru_cache(maxsize=1)
def _mbpp_groundtruth() -> tuple[dict[str, Any], dict[str, Any]]:
    """Memoized: (problems, expected_output) for MBPP+.

    ``get_groundtruth`` is expensive (runs canonical solutions on every
    input to derive expected outputs); we cache it for the process.
    """
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.evaluate import MBPP_OUTPUT_NOT_NONE_TASKS, get_groundtruth

    problems = get_mbpp_plus()
    expected = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
    return problems, expected


def _score_mbpp_plus(task: Task, code: str) -> ScoreResult:
    _apply_macos_compat()
    from evalplus.evaluate import check_correctness

    problems, expected = _mbpp_groundtruth()
    if task.id not in problems:
        raise ValueError(f"{task.id!r} not in MBPP+; cannot grade via EvalPlus.")

    result = check_correctness(
        dataset="mbpp",
        completion_id=0,
        problem=problems[task.id],
        solution=code,
        expected_output=expected[task.id],
        base_only=False,
        identifier=task.id,
    )

    plus_status, plus_results = _unpack(result.get("plus"))
    base_status, base_results = _unpack(result.get("base"))

    plus_passed = sum(1 for r in plus_results if r is True)
    plus_total = len(plus_results)
    base_passed = sum(1 for r in base_results if r is True)
    base_total = len(base_results)

    errors: list[str] = []
    if plus_status not in ("base only", "pass") and plus_total == 0:
        errors.append(f"plus status: {plus_status}")
    if plus_status not in ("pass",) and plus_total > 0:
        errors.append(f"plus status: {plus_status}")

    return ScoreResult(
        pass_count=plus_passed,
        total_count=plus_total,
        pass_rate=(plus_passed / plus_total) if plus_total > 0 else 0.0,
        base_pass_rate=(base_passed / base_total) if base_total > 0 else None,
        errors=errors,
    )


def _unpack(entry: Any) -> tuple[str, list[Any]]:
    """``check_correctness`` returns either (status, per-input-list) or None."""
    if entry is None:
        return "missing", []
    status, results = entry
    return status, list(results) if results is not None else []


# ---- Local exec for seed tasks ----


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
