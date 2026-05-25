"""Tests for batch-baseline result classification."""

from __future__ import annotations

import json
from threading import Lock

from scripts import batch_baseline
from scripts.batch_baseline import Counters, _record


def _result(task_id: str, direct_passes: int, spec_passes: int) -> dict[str, object]:
    total = 10
    return {
        "task_id": task_id,
        "direct": {
            "pass_count": direct_passes,
            "total_count": total,
            "passed_all": direct_passes == total,
        },
        "spec_guided": {
            "pass_count": spec_passes,
            "total_count": total,
            "passed_all": spec_passes == total,
        },
    }


def _score(passed_all: bool) -> dict[str, object]:
    return {
        "pass_count": 1 if passed_all else 0,
        "total_count": 1,
        "pass_rate": 1.0 if passed_all else 0.0,
        "base_pass_rate": None,
        "errors": [],
        "passed_all": passed_all,
    }


def test_record_tracks_aggregate_partial_scores(tmp_path) -> None:
    counters = Counters()
    lock = Lock()
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    meta = {"task_count": 2}

    _record(
        _result("t1", direct_passes=4, spec_passes=7),
        counters=counters,
        lock=lock,
        results_path=results_path,
        summary_path=summary_path,
        meta=meta,
    )
    _record(
        _result("t2", direct_passes=5, spec_passes=3),
        counters=counters,
        lock=lock,
        results_path=results_path,
        summary_path=summary_path,
        meta=meta,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["direct_pass_count"] == 9
    assert summary["direct_total_count"] == 20
    assert summary["spec_pass_count"] == 10
    assert summary["spec_total_count"] == 20
    assert summary["spec_minus_direct_pass_count"] == 1
    assert summary["direct_pass_rate"] == 0.45
    assert summary["spec_pass_rate"] == 0.5
    assert summary["partial_improved"] == 1
    assert summary["partial_regressed"] == 1


def test_clarified_spec_lane_defines_rescue_for_hardened_tasks(tmp_path) -> None:
    counters = batch_baseline.Counters()
    result = {
        "task_id": "hardened/x",
        "direct": _score(False),
        "spec_guided": _score(False),
        "clarified_spec": _score(True),
        "gold_spec": _score(True),
        "rescue_lane": "clarified_spec",
    }

    batch_baseline._record(
        result,
        counters=counters,
        lock=Lock(),
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        meta={"task_count": 1},
    )

    assert counters.direct_failed == 1
    assert counters.spec_failed == 1
    assert counters.clarified_failed == 0
    assert counters.gold_failed == 0
    assert counters.rescued == 1
    assert counters.generated_rescued == 0
    assert counters.gold_rescued == 1
    assert counters.both_failed == 0


def test_generated_spec_rescue_is_tracked_separately(tmp_path) -> None:
    counters = batch_baseline.Counters()
    result = {
        "task_id": "hardened/x",
        "direct": _score(False),
        "spec_guided": _score(True),
        "clarified_spec": _score(True),
        "gold_spec": _score(True),
        "rescue_lane": "clarified_spec",
    }

    batch_baseline._record(
        result,
        counters=counters,
        lock=Lock(),
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        meta={"task_count": 1},
    )

    assert counters.rescued == 1
    assert counters.generated_rescued == 1


def test_gold_spec_success_is_only_upper_bound_when_clarification_fails(tmp_path) -> None:
    counters = batch_baseline.Counters()
    result = {
        "task_id": "hardened/x",
        "direct": _score(False),
        "spec_guided": _score(False),
        "clarified_spec": _score(False),
        "gold_spec": _score(True),
        "rescue_lane": "clarified_spec",
    }

    batch_baseline._record(
        result,
        counters=counters,
        lock=Lock(),
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        meta={"task_count": 1},
    )

    assert counters.rescued == 0
    assert counters.clarified_failed == 1
    assert counters.gold_rescued == 1
    assert counters.both_failed == 1
