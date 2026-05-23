"""Tests for benchmark scoring."""

from __future__ import annotations

from idcs.benchmark.scoring import score, score_detailed
from idcs.schemas import Task, Test


def _task(tests: list[str], entry_point: str = "add") -> Task:
    return Task(
        id="test/1",
        prompt="test",
        entry_point=entry_point,
        tests=[Test(id=f"test/1/t{i}", code=t) for i, t in enumerate(tests)],
    )


class TestScore:
    def test_correct_code_returns_1(self):
        task = _task(["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
        code = "def add(a, b): return a + b"
        assert score(task, code) == 1.0

    def test_partially_correct_returns_fraction(self):
        task = _task(["assert add(1, 2) == 3", "assert add(1, 1) == 3"])
        code = "def add(a, b): return a + b"
        assert score(task, code) == 0.5

    def test_all_wrong_returns_0(self):
        task = _task(["assert add(1, 2) == 99", "assert add(0, 0) == 99"])
        code = "def add(a, b): return a + b"
        assert score(task, code) == 0.0

    def test_syntax_error_returns_0(self):
        task = _task(["assert add(1, 2) == 3"])
        code = "def add(a b): return a + b"  # missing comma
        assert score(task, code) == 0.0

    def test_runtime_error_returns_0(self):
        task = _task(["assert add(1, 2) == 3"])
        code = "def add(a, b): return a / 0"
        assert score(task, code) == 0.0

    def test_empty_tests_returns_1(self):
        task = _task([])
        assert score(task, "def f(): pass") == 1.0


class TestScoreDetailed:
    def test_reports_errors(self):
        task = _task(["assert add(1, 2) == 3", "assert add(1, 1) == 99"])
        code = "def add(a, b): return a + b"
        result = score_detailed(task, code)
        assert result.pass_count == 1
        assert result.total_count == 2
        assert len(result.errors) == 1
        assert "t1" in result.errors[0]

    def test_syntax_error_in_errors(self):
        task = _task(["assert f() == 1"])
        result = score_detailed(task, "def f( return 1")
        assert result.pass_count == 0
        assert "SyntaxError" in result.errors[0]
