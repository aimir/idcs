"""Probe direct-codegen hardness for benchmark selection.

This is intentionally direct-only: before spending calls on the full
direct-vs-spec loop, find tasks where the current model is not already perfect.

Example:
    IDCS_BACKEND=codex IDCS_CODEX_MODEL=gpt-5.4-mini \\
      uv run python scripts/probe_hardness.py --dataset hard
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idcs.benchmark.scoring import score_detailed  # noqa: E402
from idcs.benchmark.tasks import HARD_DATASET, MBPP_PLUS_DATASET, load_benchmark_tasks  # noqa: E402
from idcs.coder import Coder  # noqa: E402
from idcs.llm import LLM  # noqa: E402
from idcs.schemas import Task  # noqa: E402


@dataclass
class ProbeCounters:
    done: int = 0
    errors: int = 0
    solved: int = 0
    base_ok_plus_fail: int = 0
    partial: int = 0
    failed: int = 0


def _select_tasks(args: argparse.Namespace) -> list[Task]:
    tasks = load_benchmark_tasks(args.dataset)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [task for task in tasks if task.id in wanted]
    if args.offset:
        tasks = tasks[args.offset :]
    if args.sample:
        rng = random.Random(args.seed)
        tasks = rng.sample(tasks, min(args.sample, len(tasks)))
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def _score_payload(result: Any) -> dict[str, Any]:
    return {
        "pass_count": result.pass_count,
        "total_count": result.total_count,
        "plus_pass_rate": result.pass_rate,
        "base_pass_rate": result.base_pass_rate,
        "errors": result.errors,
        "passed_all": result.total_count > 0 and result.pass_count == result.total_count,
    }


def classify_score(*, base_pass_rate: float | None, plus_pass_rate: float) -> str:
    """Bucket a direct-only result by benchmark-selection value."""
    if plus_pass_rate >= 1.0:
        return "solved"
    if base_pass_rate is not None and base_pass_rate >= 1.0:
        return "base_ok_plus_fail"
    if plus_pass_rate > 0.0:
        return "partial"
    return "failed"


def _run_one(task: Task) -> dict[str, Any]:
    started = time.time()
    llm = LLM()
    code = Coder(llm).from_prompt(task.prompt)
    score = score_detailed(task, code)
    score_payload = _score_payload(score)
    label = classify_score(
        base_pass_rate=score_payload["base_pass_rate"],
        plus_pass_rate=score_payload["plus_pass_rate"],
    )
    return {
        "task_id": task.id,
        "entry_point": task.entry_point,
        "label": label,
        "direct": score_payload,
        "duration_s": round(time.time() - started, 3),
        "llm_calls_made": llm.calls_made,
        "generated_code": code,
    }


def _run_with_error_capture(task: Task) -> dict[str, Any]:
    try:
        return _run_one(task)
    except Exception as exc:  # noqa: BLE001 - probe should keep going.
        return {
            "task_id": task.id,
            "entry_point": task.entry_point,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def _record_result(
    result: dict[str, Any],
    *,
    counters: ProbeCounters,
    results_path: Path,
    summary_path: Path,
    meta: dict[str, Any],
) -> None:
    counters.done += 1
    if result.get("error"):
        counters.errors += 1
    else:
        label = str(result["label"])
        if label == "solved":
            counters.solved += 1
        elif label == "base_ok_plus_fail":
            counters.base_ok_plus_fail += 1
        elif label == "partial":
            counters.partial += 1
        else:
            counters.failed += 1

    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")

    # Refresh summary.json after every task so interrupted probes still
    # leave a usable summary on disk — same pattern as batch_baseline.py.
    summary = {
        **meta,
        **asdict(counters),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if result.get("error"):
        print(f"[{counters.done}] ERROR {result.get('task_id')}: {result['error']}", flush=True)
        return

    direct = result["direct"]
    print(
        f"[{counters.done}] {result['label']:<17} {result['task_id']:<9} "
        f"base={direct['base_pass_rate']} plus={direct['plus_pass_rate']:.3f} "
        f"calls={result['llm_calls_made']} {result['duration_s']:.1f}s",
        flush=True,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[MBPP_PLUS_DATASET, HARD_DATASET],
        default=HARD_DATASET,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    tasks = _select_tasks(args)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    run_dir = args.run_dir or (
        Path("experiments/runs")
        / f"{args.dataset}-hardness-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    meta = {
        "dataset": args.dataset,
        "task_count": len(tasks),
        "workers": args.workers,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"run_dir={run_dir}", flush=True)
    counters = ProbeCounters()
    if args.workers <= 1:
        for task in tasks:
            _record_result(
                _run_with_error_capture(task),
                counters=counters,
                results_path=results_path,
                summary_path=summary_path,
                meta=meta,
            )
    else:
        lock = Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_with_error_capture, task) for task in tasks]
            for future in as_completed(futures):
                with lock:
                    _record_result(
                        future.result(),
                        counters=counters,
                        results_path=results_path,
                        summary_path=summary_path,
                        meta=meta,
                    )

    print(summary_path.read_text(encoding="utf-8"), flush=True)
    return 0 if counters.errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
