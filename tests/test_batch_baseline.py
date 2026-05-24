from __future__ import annotations

import json
from threading import Lock

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
