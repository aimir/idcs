"""Tests for benchmark scoring (dispatch + both backends)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from idcs.benchmark.scoring import score, score_detailed
from idcs.schemas import Task, Test


def _seed_task(tests: list[str], entry_point: str = "add") -> Task:
    """Build a seed/* task — routes to the local exec scorer."""
    return Task(
        id="seed/test",
        prompt="test",
        entry_point=entry_point,
        tests=[Test(id=f"seed/test/t{i}", code=t) for i, t in enumerate(tests)],
    )


def _mbpp_task() -> Task:
    """Build an Mbpp/* task — routes to the EvalPlus scorer."""
    return Task(id="Mbpp/2", prompt="", entry_point="add", tests=[])


# ---- Local exec backend (seed tasks) ----


class TestLocalScorer:
    def test_correct_code_returns_1(self) -> None:
        task = _seed_task(["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
        assert score(task, "def add(a, b): return a + b") == 1.0

    def test_partially_correct_returns_fraction(self) -> None:
        task = _seed_task(["assert add(1, 2) == 3", "assert add(1, 1) == 3"])
        assert score(task, "def add(a, b): return a + b") == 0.5

    def test_all_wrong_returns_0(self) -> None:
        task = _seed_task(["assert add(1, 2) == 99"])
        assert score(task, "def add(a, b): return a + b") == 0.0

    def test_syntax_error_returns_0(self) -> None:
        task = _seed_task(["assert add(1, 2) == 3"])
        assert score(task, "def add(a b): return a + b") == 0.0

    def test_runtime_error_returns_0(self) -> None:
        task = _seed_task(["assert add(1, 2) == 3"])
        assert score(task, "def add(a, b): return a / 0") == 0.0

    def test_empty_tests_returns_1(self) -> None:
        assert score(_seed_task([]), "def f(): pass") == 1.0

    def test_detailed_reports_errors(self) -> None:
        task = _seed_task(["assert add(1, 2) == 3", "assert add(1, 1) == 99"])
        result = score_detailed(task, "def add(a, b): return a + b")
        assert result.pass_count == 1
        assert result.total_count == 2
        assert len(result.errors) == 1
        assert "t1" in result.errors[0]
        assert result.base_pass_rate is None  # local backend doesn't set this


# ---- EvalPlus backend (MBPP+ tasks) ----


class TestMbppPlusScorer:
    def _patch_groundtruth(self) -> Any:  # type: ignore[no-untyped-def]
        problems = {"Mbpp/2": {"task_id": "Mbpp/2", "entry_point": "add"}}
        expected: dict[str, Any] = {"Mbpp/2": {}}
        return patch(
            "idcs.benchmark.scoring._mbpp_groundtruth",
            return_value=(problems, expected),
        )

    def test_all_plus_pass(self) -> None:
        with self._patch_groundtruth(), patch(
            "evalplus.evaluate.check_correctness",
            return_value={
                "base": ("pass", [True, True]),
                "plus": ("pass", [True, True, True]),
            },
        ):
            result = score_detailed(_mbpp_task(), "def add(a,b): return a+b")
        assert result.pass_rate == 1.0
        assert result.pass_count == 3
        assert result.total_count == 3
        assert result.base_pass_rate == 1.0

    def test_plus_partial(self) -> None:
        with self._patch_groundtruth(), patch(
            "evalplus.evaluate.check_correctness",
            return_value={
                "base": ("pass", [True, True]),
                "plus": ("fail", [True, False, False, True]),
            },
        ):
            result = score_detailed(_mbpp_task(), "def add(a,b): return a+b")
        # Pattern-match signature: base full, plus partial.
        assert result.pass_rate == 0.5
        assert result.base_pass_rate == 1.0
        assert any("plus status" in e for e in result.errors)

    def test_missing_task_raises(self) -> None:
        with patch(
            "idcs.benchmark.scoring._mbpp_groundtruth",
            return_value=({}, {}),
        ):
            try:
                score_detailed(_mbpp_task(), "def add(a,b): return a+b")
            except ValueError as e:
                assert "Mbpp/2" in str(e)
            else:
                raise AssertionError("expected ValueError")
