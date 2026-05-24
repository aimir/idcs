"""Run direct-vs-spec baselines over MBPP+ with parallel task workers.

This is the same comparison as ``scripts/baseline.py``:

- direct: task.prompt -> code
- spec_guided: task.prompt -> G/D spec loop -> code

The difference is operational: results are written incrementally to JSONL so
long hackathon runs can be inspected while still running or after interruption.

Example:
    IDCS_BACKEND=codex IDCS_CODEX_MODEL=gpt-5.4-mini \\
      uv run python scripts/batch_baseline.py --workers 4 --limit 50
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from threading import Lock
from typing import Any

from idcs.benchmark.scoring import score_detailed  # noqa: E402
from idcs.benchmark.tasks import load_mbpp_plus  # noqa: E402
from idcs.coder import Coder  # noqa: E402
from idcs.distinguisher import Distinguisher  # noqa: E402
from idcs.generator import Generator  # noqa: E402
from idcs.llm import LLM  # noqa: E402
from idcs.orchestrator import run_episode  # noqa: E402
from idcs.schemas import Task  # noqa: E402
from idcs.user_proxy import NullUserProxy  # noqa: E402


@dataclass
class Counters:
    done: int = 0
    errors: int = 0
    direct_failed: int = 0
    spec_failed: int = 0
    rescued: int = 0
    regressed: int = 0
    both_failed: int = 0


def _select_tasks(args: argparse.Namespace) -> list[Task]:
    tasks = load_mbpp_plus()
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
        "pass_rate": result.pass_rate,
        "base_pass_rate": result.base_pass_rate,
        "errors": result.errors,
        "passed_all": result.total_count > 0 and result.pass_count == result.total_count,
    }


def _run_one(task: Task, *, max_turns: int) -> dict[str, Any]:
    started = time.time()
    llm = LLM()
    coder = Coder(llm)
    generator = Generator(llm)
    distinguisher = Distinguisher(llm)

    direct_code = coder.from_prompt(task.prompt)
    direct_score = score_detailed(task, direct_code)

    trace = run_episode(task, generator, distinguisher, NullUserProxy(), max_turns=max_turns)
    if trace.final_spec is None:
        raise RuntimeError("no final spec produced")
    spec_code = coder.from_spec(trace.final_spec, task.prompt)
    spec_score = score_detailed(task, spec_code)

    return {
        "task_id": task.id,
        "entry_point": task.entry_point,
        "direct": _score_payload(direct_score),
        "spec_guided": _score_payload(spec_score),
        "turn_count": len(trace.turns),
        "issue_count": sum(len(turn.issues) for turn in trace.turns),
        "duration_s": round(time.time() - started, 3),
        "llm_calls_made": llm.calls_made,
    }


def _run_with_retries(task: Task, *, max_turns: int, retries: int) -> dict[str, Any]:
    errors: list[dict[str, str | int]] = []
    for attempt in range(retries + 1):
        try:
            result = _run_one(task, max_turns=max_turns)
            result["attempts"] = attempt + 1
            return result
        except Exception as exc:  # noqa: BLE001 - batch runner should keep going.
            errors.append({"attempt": attempt + 1, "error": repr(exc)})
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
    return {
        "task_id": task.id,
        "entry_point": task.entry_point,
        "error": errors[-1]["error"],
        "attempt_errors": errors,
        "traceback": traceback.format_exc(limit=8),
    }


def _record(
    result: dict[str, Any],
    *,
    counters: Counters,
    lock: Lock,
    results_path: Path,
    summary_path: Path,
    meta: dict[str, Any],
) -> None:
    with lock:
        counters.done += 1
        if result.get("error"):
            counters.errors += 1
        else:
            direct_ok = bool(result["direct"]["passed_all"])
            spec_ok = bool(result["spec_guided"]["passed_all"])
            if not direct_ok:
                counters.direct_failed += 1
            if not spec_ok:
                counters.spec_failed += 1
            if not direct_ok and spec_ok:
                counters.rescued += 1
            if direct_ok and not spec_ok:
                counters.regressed += 1
            if not direct_ok and not spec_ok:
                counters.both_failed += 1

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")

        summary = {
            **meta,
            **asdict(counters),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{counters.done}/{meta['task_count']}] "
            f"direct_failed={counters.direct_failed} "
            f"spec_failed={counters.spec_failed} "
            f"rescued={counters.rescued} "
            f"regressed={counters.regressed} "
            f"errors={counters.errors} "
            f"task={result.get('task_id')} "
            f"attempts={result.get('attempts', 'ERR')}",
            flush=True,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=3)
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
        / f"mbpp-plus-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    meta = {
        "dataset": "mbpp-plus",
        "task_count": len(tasks),
        "workers": args.workers,
        "retries": args.retries,
        "max_turns": args.max_turns,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"run_dir={run_dir}", flush=True)
    print(
        f"tasks={len(tasks)} workers={args.workers} retries={args.retries} "
        f"max_turns={args.max_turns}",
        flush=True,
    )

    counters = Counters()
    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_with_retries,
                task,
                max_turns=args.max_turns,
                retries=args.retries,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            _record(
                result,
                counters=counters,
                lock=lock,
                results_path=results_path,
                summary_path=summary_path,
                meta=meta,
            )

    print(summary_path.read_text(encoding="utf-8"), flush=True)
    return 0 if counters.errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
