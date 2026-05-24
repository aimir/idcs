"""Tests for the EvalPlus task-loading adapter.

EvalPlus is mocked so the tests don't hit the dataset cache / network.
"""

from __future__ import annotations

from unittest.mock import patch

from idcs.benchmark.tasks import load_mbpp_plus


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
