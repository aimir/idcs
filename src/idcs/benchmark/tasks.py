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

from idcs.benchmark.scoring import EQUALITY_HELPER_SRC, INFINITY_LITERAL_PRELUDE
from idcs.schemas import Task, Test

MBPP_PLUS_DATASET = "mbpp-plus"
HARD_DATASET = "hard"

# Small MBPP+ slice chosen by direct-only probing for underspecified edge
# semantics rather than algorithmic difficulty. These are tasks where
# gpt-5.4-mini was not already perfect, so a spec loop has room to matter.
HARD_MBPP_PLUS_IDS: tuple[str, ...] = (
    "Mbpp/639",  # count only proper-cased names, not every non-lowercase start
    "Mbpp/427",  # rewrite date-shaped text, including non-calendar dates
    "Mbpp/459",  # keep lowercase letters only, not punctuation or non-uppercase chars
    "Mbpp/92",  # undulating means exactly two alternating digits, including short cases
    "Mbpp/597",  # kth merged element with empty arrays and mixed comparable values
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
        return load_mbpp_plus(max_plus_inputs=max_plus_inputs)
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
    # ``repr(cases)`` emits bare ``inf`` / ``nan`` for infinity/NaN values;
    # ``INFINITY_LITERAL_PRELUDE`` defines those names so the literal parses.
    # ``EQUALITY_HELPER_SRC`` is shared with ``scoring._HARNESS``.
    return (
        "import math\n\n"
        f"{INFINITY_LITERAL_PRELUDE}\n"
        f"_IDCS_ORACLE_SRC = {canonical_solution!r}\n"
        "_IDCS_ORACLE_NS = {}\n"
        "exec(_IDCS_ORACLE_SRC, _IDCS_ORACLE_NS)\n"
        f"_IDCS_ORACLE = _IDCS_ORACLE_NS[{entry_point!r}]\n"
        f"_IDCS_CANDIDATE = {entry_point}\n"
        f"_IDCS_CASES = {list(cases)!r}\n"
        f"_IDCS_ATOL = {atol!r}\n\n"
        f"{EQUALITY_HELPER_SRC}\n"
        "for _IDCS_CASE in _IDCS_CASES:\n"
        "    _IDCS_ACTUAL = _IDCS_CANDIDATE(*_IDCS_CASE)\n"
        "    _IDCS_EXPECTED = _IDCS_ORACLE(*_IDCS_CASE)\n"
        "    assert _idcs_equal(_IDCS_ACTUAL, _IDCS_EXPECTED, _IDCS_ATOL), (\n"
        "        f'input={_IDCS_CASE!r} expected={_IDCS_EXPECTED!r} actual={_IDCS_ACTUAL!r}'\n"
        "    )\n"
    )
