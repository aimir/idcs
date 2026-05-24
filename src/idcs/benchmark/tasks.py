"""Task-loading adapter for the EvalPlus benchmark library.

EvalPlus's MBPP+ shape is normalized into our ``Task`` / ``Test`` schema so
the rest of the pipeline (G, D, orchestrator) sees a single Task type
regardless of source. Scoring will route back through EvalPlus in a
follow-up — see ``benchmark/scoring.py`` (TODO Phase 2).

EvalPlus is imported lazily so that importing this module does not drag in
its full dependency tree (numpy, datasets, transformers, ...).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from idcs.schemas import Task, Test

MBPP_PLUS_DATASET = "mbpp-plus"
HARD_DATASET = "hard"

# Small MBPP+ slice chosen for underspecified edge semantics rather than
# algorithmic difficulty. These are the tasks where a better spec should matter.
HARD_MBPP_PLUS_IDS: tuple[str, ...] = (
    "Mbpp/459",  # remove uppercase characters, including empty/all-uppercase strings
    "Mbpp/11",  # remove first and last occurrence only
    "Mbpp/100",  # next palindrome: 0/1/9/carry/already-palindrome cases
    "Mbpp/733",  # leftmost duplicate in sorted arrays
    "Mbpp/580",  # recursively preserve nested tuple structure while filtering
    "Mbpp/572",  # keep values appearing exactly once, not ordinary dedupe
    "Mbpp/758",  # tuple-key list counting, including empty sublists
    "Mbpp/94",  # first tuple field at the minimum second field
    "Mbpp/167",  # 0 -> 1 and exact powers of two
)

HUMANEVAL_PLUS_FOLLOWUP_IDS: tuple[str, ...] = (
    "HumanEval/125",
    "HumanEval/126",
    "HumanEval/99",
    "HumanEval/124",
    "HumanEval/26",
    "HumanEval/95",
    "HumanEval/134",
    "HumanEval/68",
)


def load_mbpp_plus(
    *,
    task_ids: Sequence[str] | None = None,
    include_evalplus_tests: bool = False,
    max_plus_inputs: int | None = None,
) -> list[Task]:
    """Return every MBPP+ task, normalized to our ``Task`` schema."""
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    selected_ids = list(task_ids) if task_ids is not None else list(problems)
    tasks: list[Task] = []
    for task_id in selected_ids:
        if task_id not in problems:
            raise ValueError(f"{task_id!r} is not available in MBPP+.")
        tasks.append(
            _to_task(
                task_id,
                problems[task_id],
                include_evalplus_tests=include_evalplus_tests,
                max_plus_inputs=max_plus_inputs,
            )
        )
    return tasks


def load_mbpp_hard(*, max_plus_inputs: int | None = None) -> list[Task]:
    """Return the named MBPP+ hard slice for spec-sensitivity experiments."""
    return load_mbpp_plus(
        task_ids=HARD_MBPP_PLUS_IDS,
        include_evalplus_tests=True,
        max_plus_inputs=max_plus_inputs,
    )


def load_benchmark_tasks(
    dataset: str,
    *,
    max_plus_inputs: int | None = None,
) -> list[Task]:
    """Load a named benchmark dataset used by the scripts."""
    if dataset in {"mbpp", MBPP_PLUS_DATASET}:
        return load_mbpp_plus()
    if dataset == HARD_DATASET:
        return load_mbpp_hard(max_plus_inputs=max_plus_inputs)
    raise ValueError(
        f"Unknown dataset {dataset!r}; expected '{MBPP_PLUS_DATASET}' or '{HARD_DATASET}'."
    )


def _to_task(
    task_id: str,
    item: dict[str, Any],
    *,
    include_evalplus_tests: bool = False,
    max_plus_inputs: int | None = None,
) -> Task:
    prompt = str(item.get("prompt", ""))
    entry_point = item.get("entry_point") or None
    if include_evalplus_tests:
        tests = _evalplus_tests(task_id, item, max_plus_inputs=max_plus_inputs)
    else:
        assertion_block = str(item.get("assertion", ""))
        tests = [
            Test(id=f"{task_id}/t{i}", code=line)
            for i, line in enumerate(_split_assertions(assertion_block))
        ]
    return Task(id=task_id, prompt=prompt, entry_point=entry_point, tests=tests)


def _split_assertions(text: str) -> list[str]:
    """Pull out each ``assert ...`` line from a multi-assert block."""
    return [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()).startswith("assert")
    ]


def _evalplus_tests(
    task_id: str,
    item: dict[str, Any],
    *,
    max_plus_inputs: int | None,
) -> list[Test]:
    entry_point = str(item.get("entry_point") or "")
    canonical_solution = str(item.get("canonical_solution") or "")
    if not entry_point or not canonical_solution:
        raise ValueError(f"{task_id!r} is missing EvalPlus entry point or solution.")

    atol = float(item.get("atol") or 0.0)
    base_inputs = list(item.get("base_input") or [])
    plus_inputs = list(item.get("plus_input") or [])
    if max_plus_inputs is not None:
        plus_inputs = plus_inputs[:max_plus_inputs]

    tests: list[Test] = []
    if base_inputs:
        tests.append(
            Test(
                id=f"{task_id}/base-inputs",
                code=_build_evalplus_case_test(
                    entry_point,
                    canonical_solution,
                    base_inputs,
                    atol,
                ),
            )
        )
    if plus_inputs:
        tests.append(
            Test(
                id=f"{task_id}/plus-inputs",
                code=_build_evalplus_case_test(
                    entry_point,
                    canonical_solution,
                    plus_inputs,
                    atol,
                ),
            )
        )
    return tests


def _build_evalplus_case_test(
    entry_point: str,
    canonical_solution: str,
    cases: Sequence[Any],
    atol: float,
) -> str:
    return (
        "import math\n\n"
        f"_IDCS_ORACLE_SRC = {canonical_solution!r}\n"
        "_IDCS_ORACLE_NS = {}\n"
        "exec(_IDCS_ORACLE_SRC, _IDCS_ORACLE_NS)\n"
        f"_IDCS_ORACLE = _IDCS_ORACLE_NS[{entry_point!r}]\n"
        f"_IDCS_CANDIDATE = {entry_point}\n"
        f"_IDCS_CASES = {list(cases)!r}\n"
        f"_IDCS_ATOL = {atol!r}\n\n"
        f"{_evalplus_equality_helper()}\n\n"
        "for _IDCS_CASE in _IDCS_CASES:\n"
        "    _IDCS_ACTUAL = _IDCS_CANDIDATE(*_IDCS_CASE)\n"
        "    _IDCS_EXPECTED = _IDCS_ORACLE(*_IDCS_CASE)\n"
        "    assert _idcs_equal(_IDCS_ACTUAL, _IDCS_EXPECTED, _IDCS_ATOL), (\n"
        "        f'input={_IDCS_CASE!r} expected={_IDCS_EXPECTED!r} actual={_IDCS_ACTUAL!r}'\n"
        "    )\n"
    )


def _evalplus_equality_helper() -> str:
    return """
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
""".strip()
