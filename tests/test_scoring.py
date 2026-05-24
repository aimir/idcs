"""Tests for benchmark scoring (dispatch + both backends)."""

from __future__ import annotations

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
    """Build an Mbpp/* task — routes to the EvalPlus-data + subprocess scorer."""
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


# ---- MBPP+ backend (EvalPlus data + our subprocess) ----


class TestMbppPlusScorer:
    """The subprocess seam is ``_run_grader`` — mock it to skip real exec."""

    def _problems(self) -> dict[str, dict[str, object]]:
        return {
            "Mbpp/2": {
                "task_id": "Mbpp/2",
                "entry_point": "add",
                "plus_input": [[1, 2], [3, 4], [5, 6]],
                "base_input": [[0, 0], [1, 1]],
            }
        }

    def _expected(self) -> dict[str, dict[str, list[object]]]:
        return {"Mbpp/2": {"plus": [3, 7, 11], "base": [0, 2]}}

    def test_all_plus_pass(self) -> None:
        with patch(
            "idcs.benchmark.scoring._mbpp_groundtruth",
            return_value=(self._problems(), self._expected()),
        ), patch(
            "idcs.benchmark.scoring._run_grader",
            side_effect=[
                ([True, True, True], None),  # plus
                ([True, True], None),  # base
            ],
        ):
            result = score_detailed(_mbpp_task(), "def add(a,b): return a+b")
        assert result.pass_rate == 1.0
        assert result.pass_count == 3
        assert result.total_count == 3
        assert result.base_pass_rate == 1.0
        assert result.errors == []

    def test_plus_partial_base_full_signals_pattern_match(self) -> None:
        with patch(
            "idcs.benchmark.scoring._mbpp_groundtruth",
            return_value=(self._problems(), self._expected()),
        ), patch(
            "idcs.benchmark.scoring._run_grader",
            side_effect=[
                ([True, False, False], None),  # plus
                ([True, True], None),  # base
            ],
        ):
            result = score_detailed(_mbpp_task(), "def add(a,b): return a+b")
        assert result.pass_rate == 1 / 3
        assert result.base_pass_rate == 1.0

    def test_subprocess_error_surfaces(self) -> None:
        with patch(
            "idcs.benchmark.scoring._mbpp_groundtruth",
            return_value=(self._problems(), self._expected()),
        ), patch(
            "idcs.benchmark.scoring._run_grader",
            side_effect=[
                ([False, False, False], "subprocess timeout"),
                ([False, False], "subprocess timeout"),
            ],
        ):
            result = score_detailed(_mbpp_task(), "while True: pass")
        assert result.pass_rate == 0.0
        assert any("timeout" in e for e in result.errors)

    def test_missing_task_raises(self) -> None:
        with patch("idcs.benchmark.scoring._mbpp_groundtruth", return_value=({}, {})):
            try:
                score_detailed(_mbpp_task(), "def add(a,b): return a+b")
            except ValueError as e:
                assert "Mbpp/2" in str(e)
            else:
                raise AssertionError("expected ValueError")


class TestRunGraderEndToEnd:
    """Cross-check that the actual subprocess harness works (no mock)."""

    def test_correct_code_passes_all_inputs(self) -> None:
        from idcs.benchmark.scoring import _run_grader

        results, err = _run_grader(
            code="def add(a, b):\n    return a + b\n",
            entry="add",
            inputs=[[1, 2], [3, 4], [10, 20]],
            expected=[3, 7, 30],
        )
        assert err is None
        assert results == [True, True, True]

    def test_wrong_code_fails_all(self) -> None:
        from idcs.benchmark.scoring import _run_grader

        results, err = _run_grader(
            code="def add(a, b):\n    return 0\n",
            entry="add",
            inputs=[[1, 2], [3, 4]],
            expected=[3, 7],
        )
        assert err is None
        assert results == [False, False]

    def test_missing_entry_point_fails_all(self) -> None:
        from idcs.benchmark.scoring import _run_grader

        results, err = _run_grader(
            code="def other(): return 1\n",
            entry="add",
            inputs=[[1, 2]],
            expected=[3],
        )
        assert err is None
        assert results == [False]

    def test_tuple_inputs_unpack_correctly(self) -> None:
        from idcs.benchmark.scoring import _run_grader

        # MBPP+ inputs are often nested tuples; repr round-trips them.
        results, err = _run_grader(
            code="def f(a, b):\n    return tuple(sorted(set(a) & set(b)))\n",
            entry="f",
            inputs=[[(3, 4, 5, 6), (5, 7, 4, 10)]],
            expected=[(4, 5)],
        )
        assert err is None
        assert results == [True]
