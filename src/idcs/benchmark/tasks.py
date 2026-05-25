"""Task-loading adapter for the EvalPlus benchmark library.

EvalPlus's MBPP+ shape is normalized into our ``Task`` / ``Test`` schema so
the rest of the pipeline (G, D, orchestrator) sees a single Task type
regardless of source. Scoring will route back through EvalPlus in a
follow-up — see ``benchmark/scoring.py`` (TODO Phase 2).

EvalPlus is imported lazily so that importing this module does not drag in
its full dependency tree (numpy, datasets, transformers, ...).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idcs.benchmark.scoring import EQUALITY_HELPER_SRC, INFINITY_LITERAL_PRELUDE
from idcs.schemas import Spec, Task, Test

MBPP_PLUS_DATASET = "mbpp-plus"
HARD_DATASET = "hard"
HARD_EXTENDED_DATASET = "hard-extended"
HARD_TRAIN_DATASET = "hard-train"
HARD_DEV_DATASET = "hard-dev"
HARD_TEST_DATASET = "hard-test"
HARDENED_DATASET = "hardened"
DEFAULT_HARDENED_DIR = Path(__file__).resolve().parents[3] / "data" / "hardened_tasks"

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

# Generalization split. The original 5-task hard slice stays as the training
# surface for prompt search. Dev/test are semantically similar but held out:
# simple-looking tasks with exact edge semantics, not algorithm-heavy tasks.
HARD_MBPP_TRAIN_IDS: tuple[str, ...] = HARD_MBPP_PLUS_IDS

# Fresh held-out failures from a direct-only random MBPP+ probe with gpt-5.5
# (seed=20260525, sample=60). These were not used by the hard-train prompt
# search and are the current generalization surface.
HARD_MBPP_DEV_IDS: tuple[str, ...] = (
    "Mbpp/294",  # heterogeneous max should ignore non-integers
    "Mbpp/137",  # zero/nonzero ratio with all-zero boundary
    "Mbpp/265",  # every nth element split means S[i::n], not chunks
    "Mbpp/301",  # dictionary depth recurses into nested dictionaries
)

HARD_MBPP_TEST_IDS: tuple[str, ...] = (
    "Mbpp/785",  # tuple string parsing with ellipsis/spacing quirks
    "Mbpp/451",  # only literal spaces are removed by canonical behavior
    "Mbpp/757",  # reverse-pair counting without double counting
    "Mbpp/576",  # sublist means subsequence, not contiguous slice
    "Mbpp/765",  # nth polite number formula boundary
    "Mbpp/99",  # binary string output, no leading zeros, zero boundary
    "Mbpp/473",  # tuple intersection ignores tuple element order
    "Mbpp/777",  # non-repeated means values that occur exactly once
    "Mbpp/305",  # find two p-starting words across strings, preserving order
    "Mbpp/759",  # decimal number with exactly two digits of precision
    "Mbpp/161",  # remove every element whose value appears in another list
    "Mbpp/630",  # adjacent coordinate grid includes center and fixed ordering
    "Mbpp/794",  # regex-like a...b match must cover the full string
)

HARD_MBPP_EXTENDED_IDS: tuple[str, ...] = (
    *HARD_MBPP_TRAIN_IDS,
    *HARD_MBPP_DEV_IDS,
    *HARD_MBPP_TEST_IDS,
)


@dataclass(frozen=True)
class HardenedItem:
    """One curated task whose raw prompt is weaker than its gold spec."""

    path: Path
    task: Task
    gold_spec: Spec
    known_weakness: dict[str, Any]


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
    return load_mbpp_hard_split(HARD_DATASET, max_plus_inputs=max_plus_inputs)


def load_mbpp_hard_split(
    dataset: str,
    *,
    max_plus_inputs: int | None = None,
) -> list[Task]:
    """Return a named MBPP+ hard split for generalization experiments."""
    ids_by_dataset = {
        HARD_DATASET: HARD_MBPP_PLUS_IDS,
        HARD_EXTENDED_DATASET: HARD_MBPP_EXTENDED_IDS,
        HARD_TRAIN_DATASET: HARD_MBPP_TRAIN_IDS,
        HARD_DEV_DATASET: HARD_MBPP_DEV_IDS,
        HARD_TEST_DATASET: HARD_MBPP_TEST_IDS,
    }
    try:
        task_ids = ids_by_dataset[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown hard split {dataset!r}.") from exc
    return load_mbpp_plus(
        task_ids=task_ids,
        include_evalplus_tests=True,
        max_plus_inputs=max_plus_inputs,
    )


def load_hardened_items(
    hardened_dir: Path | None = None,
) -> list[HardenedItem]:
    """Load the file-backed hardened corpus with oracle-only gold specs."""
    directory = hardened_dir or DEFAULT_HARDENED_DIR
    items: list[HardenedItem] = []
    for path in sorted(directory.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        missing = {"task", "gold_spec"} - raw.keys()
        if missing:
            raise ValueError(f"{path} missing required field(s): {', '.join(sorted(missing))}")
        known_weakness = raw.get("known_weakness", {})
        if not isinstance(known_weakness, dict):
            raise ValueError(f"{path} field 'known_weakness' must be an object when present.")
        items.append(
            HardenedItem(
                path=path,
                task=Task.model_validate(raw["task"]),
                gold_spec=Spec.model_validate(raw["gold_spec"]),
                known_weakness=known_weakness,
            )
        )
    return items


def load_hardened_tasks(
    hardened_dir: Path | None = None,
) -> list[Task]:
    """Return only the Task objects for the hardened corpus."""
    return [item.task for item in load_hardened_items(hardened_dir)]


def load_benchmark_tasks(
    dataset: str,
    *,
    hardened_dir: Path | None = None,
    max_plus_inputs: int | None = None,
) -> list[Task]:
    """Load a named benchmark dataset used by the scripts."""
    if dataset in {"mbpp", MBPP_PLUS_DATASET}:
        return load_mbpp_plus(max_plus_inputs=max_plus_inputs)
    if dataset in {
        HARD_DATASET,
        HARD_EXTENDED_DATASET,
        HARD_TRAIN_DATASET,
        HARD_DEV_DATASET,
        HARD_TEST_DATASET,
    }:
        return load_mbpp_hard_split(dataset, max_plus_inputs=max_plus_inputs)
    if dataset == HARDENED_DATASET:
        return load_hardened_tasks(hardened_dir)
    raise ValueError(
        f"Unknown dataset {dataset!r}; expected '{MBPP_PLUS_DATASET}', a hard split, "
        f"or '{HARDENED_DATASET}'."
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
