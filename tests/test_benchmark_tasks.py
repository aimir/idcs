"""Tests for the EvalPlus task-loading adapter.

EvalPlus is mocked so the tests don't hit the dataset cache / network.
"""

from __future__ import annotations

from unittest.mock import patch

from idcs.benchmark.tasks import (
    HARD_MBPP_DEV_IDS,
    HARD_MBPP_EXTENDED_IDS,
    HARD_MBPP_PLUS_IDS,
    HARD_MBPP_TEST_IDS,
    HARD_MBPP_TRAIN_IDS,
    load_benchmark_tasks,
    load_mbpp_hard,
    load_mbpp_plus,
)


def test_normalizes_to_task_schema() -> None:
    fake = {
        "Mbpp/1": {
            "task_id": "Mbpp/1",
            "prompt": '"""\nWrite a function to add two ints.\n"""\n',
            "entry_point": "add",
            "assertion": "\nassert add(1, 2) == 3\nassert add(0, 0) == 0\n",
        },
        "Mbpp/2": {
            "task_id": "Mbpp/2",
            "prompt": '"""Sum a list."""',
            "entry_point": "sum_list",
            "assertion": "assert sum_list([1, 2, 3]) == 6",
        },
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        tasks = load_mbpp_plus()

    by_id = {t.id: t for t in tasks}
    assert set(by_id) == {"Mbpp/1", "Mbpp/2"}
    assert by_id["Mbpp/1"].prompt.startswith('"""')
    assert len(by_id["Mbpp/1"].tests) == 2
    assert by_id["Mbpp/1"].tests[0].id == "Mbpp/1/t0"
    assert "add(1, 2) == 3" in by_id["Mbpp/1"].tests[0].code
    assert len(by_id["Mbpp/2"].tests) == 1


def test_no_assertions_yields_empty_test_list() -> None:
    fake = {
        "Mbpp/9": {
            "task_id": "Mbpp/9",
            "prompt": '"""Some task."""',
            "entry_point": "f",
            "assertion": "",
        }
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        tasks = load_mbpp_plus()

    assert tasks[0].tests == []


def test_skips_non_assert_lines() -> None:
    fake = {
        "Mbpp/3": {
            "task_id": "Mbpp/3",
            "prompt": "p",
            "entry_point": "f",
            "assertion": "# a comment\n\nassert f(1) == 1\nfoo()\nassert f(2) == 2\n",
        }
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        tasks = load_mbpp_plus()

    assert [t.code for t in tasks[0].tests] == ["assert f(1) == 1", "assert f(2) == 2"]


def test_load_mbpp_plus_can_keep_explicit_id_order() -> None:
    fake = {
        "Mbpp/1": _fake_evalplus_problem("one"),
        "Mbpp/2": _fake_evalplus_problem("two"),
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        tasks = load_mbpp_plus(task_ids=["Mbpp/2", "Mbpp/1"])

    assert [task.id for task in tasks] == ["Mbpp/2", "Mbpp/1"]


def test_hard_slice_loads_intended_mbpp_plus_ids() -> None:
    fake = {
        task_id: _fake_evalplus_problem(task_id.replace("/", "_"))
        for task_id in HARD_MBPP_PLUS_IDS
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        tasks = load_mbpp_hard(max_plus_inputs=2)

    assert [task.id for task in tasks] == list(HARD_MBPP_PLUS_IDS)
    assert all(task.id.startswith("Mbpp/") for task in tasks)


def test_hard_generalization_splits_are_disjoint_and_ordered() -> None:
    assert HARD_MBPP_TRAIN_IDS == HARD_MBPP_PLUS_IDS
    assert not set(HARD_MBPP_TRAIN_IDS) & set(HARD_MBPP_DEV_IDS)
    assert not set(HARD_MBPP_TRAIN_IDS) & set(HARD_MBPP_TEST_IDS)
    assert not set(HARD_MBPP_DEV_IDS) & set(HARD_MBPP_TEST_IDS)
    assert (
        *HARD_MBPP_TRAIN_IDS,
        *HARD_MBPP_DEV_IDS,
        *HARD_MBPP_TEST_IDS,
    ) == HARD_MBPP_EXTENDED_IDS


def test_hard_dev_and_test_splits_load_intended_ids() -> None:
    fake = {
        task_id: _fake_evalplus_problem(task_id.replace("/", "_"))
        for task_id in HARD_MBPP_EXTENDED_IDS
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        dev_tasks = load_benchmark_tasks("hard-dev", max_plus_inputs=2)
        test_tasks = load_benchmark_tasks("hard-test", max_plus_inputs=2)

    assert [task.id for task in dev_tasks] == list(HARD_MBPP_DEV_IDS)
    assert [task.id for task in test_tasks] == list(HARD_MBPP_TEST_IDS)


def test_hard_slice_includes_base_and_plus_hidden_tests() -> None:
    fake = {
        task_id: _fake_evalplus_problem(task_id.replace("/", "_"))
        for task_id in HARD_MBPP_PLUS_IDS
    }
    with patch("evalplus.data.get_mbpp_plus", return_value=fake):
        task = load_benchmark_tasks("hard", max_plus_inputs=2)[0]

    assert [test.id for test in task.tests] == [
        f"{task.id}/base-inputs",
        f"{task.id}/plus-inputs",
    ]
    assert "_IDCS_ORACLE_SRC" in task.tests[0].code
    assert "_IDCS_CASES = [[1], [2]]" in task.tests[0].code
    assert "_IDCS_CASES = [[3], [4]]" in task.tests[1].code
    assert "[5]" not in task.tests[1].code


def _fake_evalplus_problem(name: str) -> dict[str, object]:
    entry_point = f"identity_{name}".replace("-", "_")
    return {
        "task_id": name,
        "prompt": f'"""\nReturn the input for {name}.\n"""',
        "entry_point": entry_point,
        "assertion": f"assert {entry_point}(1) == 1",
        "canonical_solution": f"def {entry_point}(x):\n    return x\n",
        "base_input": [[1], [2]],
        "plus_input": [[3], [4], [5]],
        "atol": 0.0,
    }
