"""Score generated code against a task's test assertions.

Uses simple exec() in an isolated namespace. Not sandboxed — MBPP+ tasks
are trusted and don't do anything dangerous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from idcs.schemas import Task


@dataclass
class ScoreResult:
    """Detailed scoring result for one task."""

    pass_count: int
    total_count: int
    pass_rate: float
    errors: list[str] = field(default_factory=list)


def score(task: Task, code: str) -> float:
    """Execute code and run task assertions; return pass rate in [0, 1]."""
    return score_detailed(task, code).pass_rate


def score_detailed(task: Task, code: str) -> ScoreResult:
    """Execute code and return detailed scoring results."""
    if not task.tests:
        return ScoreResult(pass_count=0, total_count=0, pass_rate=1.0)

    namespace: dict = {}
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
