"""Run direct-vs-spec baselines over MBPP+ with parallel task workers.

This is the same comparison as ``scripts/baseline.py``:

- direct: task.prompt -> code
- spec_guided: task.prompt -> G/D spec loop -> code

The difference is operational: results are written incrementally to JSONL so
long hackathon runs can be inspected while still running or after interruption.

Example:
    IDCS_BACKEND=codex IDCS_CODEX_MODEL=gpt-5.4-mini \\
      uv run python scripts/batch_baseline.py --dataset hard --workers 4
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
_SCRIPT_DIR_TEXT = str(_SCRIPT_DIR)
while _SCRIPT_DIR_TEXT in sys.path:
    sys.path.remove(_SCRIPT_DIR_TEXT)
_SRC = _SCRIPT_DIR.parent / "src"
_SRC_TEXT = str(_SRC)
while _SRC_TEXT in sys.path:
    sys.path.remove(_SRC_TEXT)
sys.path.insert(0, _SRC_TEXT)

from idcs.benchmark.scoring import score_detailed  # noqa: E402
from idcs.benchmark.tasks import (  # noqa: E402
    HARD_DATASET,
    HARD_DEV_DATASET,
    HARD_EXTENDED_DATASET,
    HARD_TEST_DATASET,
    HARD_TRAIN_DATASET,
    HARDENED_DATASET,
    MBPP_PLUS_DATASET,
    load_benchmark_tasks,
    load_hardened_items,
)
from idcs.coder import Coder  # noqa: E402
from idcs.distinguisher import Distinguisher  # noqa: E402
from idcs.generator import Generator  # noqa: E402
from idcs.llm import LLM, runtime_snapshot  # noqa: E402
from idcs.orchestrator import run_episode  # noqa: E402
from idcs.schemas import Spec, Task  # noqa: E402
from idcs.user_proxy import NullUserProxy, OracleUserProxy  # noqa: E402


@dataclass
class Counters:
    done: int = 0
    errors: int = 0
    direct_failed: int = 0
    spec_failed: int = 0
    clarified_failed: int = 0
    gold_failed: int = 0
    rescued: int = 0
    generated_rescued: int = 0
    gold_rescued: int = 0
    regressed: int = 0
    both_failed: int = 0
    direct_pass_count: int = 0
    direct_total_count: int = 0
    spec_pass_count: int = 0
    spec_total_count: int = 0
    partial_improved: int = 0
    partial_regressed: int = 0
    partial_unchanged: int = 0


def _select_tasks(args: argparse.Namespace) -> tuple[list[Task], dict[str, Spec]]:
    if args.dataset == HARDENED_DATASET:
        hardened_items = load_hardened_items(args.hardened_dir)
        tasks = [item.task for item in hardened_items]
        gold_specs = {item.task.id: item.gold_spec for item in hardened_items}
    else:
        tasks = load_benchmark_tasks(args.dataset)
        gold_specs = {}

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
    selected_ids = {task.id for task in tasks}
    return tasks, {task_id: spec for task_id, spec in gold_specs.items() if task_id in selected_ids}


def _score_payload(result: Any) -> dict[str, Any]:
    return {
        "pass_count": result.pass_count,
        "total_count": result.total_count,
        "pass_rate": result.pass_rate,
        "base_pass_rate": result.base_pass_rate,
        "errors": result.errors,
        "failure_examples": [
            asdict(example) for example in getattr(result, "failure_examples", [])
        ],
        "passed_all": result.total_count > 0 and result.pass_count == result.total_count,
    }


def _read_prompt_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def _run_one(
    task: Task,
    *,
    max_turns: int,
    generator_prompt: str | None,
    distinguisher_prompt: str | None,
    coder_prompt: str | None,
    gold_spec: Spec | None = None,
) -> dict[str, Any]:
    started = time.time()
    llm = LLM()
    coder = Coder(llm, prompt=coder_prompt) if coder_prompt is not None else Coder(llm)
    generator = (
        Generator(llm, prompt=generator_prompt)
        if generator_prompt is not None
        else Generator(llm)
    )
    distinguisher = (
        Distinguisher(llm, prompt=distinguisher_prompt)
        if distinguisher_prompt is not None
        else Distinguisher(llm)
    )

    direct_code = coder.from_prompt(task.prompt)
    direct_score = score_detailed(task, direct_code)

    trace = run_episode(task, generator, distinguisher, NullUserProxy(), max_turns=max_turns)
    if trace.final_spec is None:
        raise RuntimeError("no final spec produced")
    spec_code = coder.from_spec(trace.final_spec, task.prompt)
    spec_score = score_detailed(task, spec_code)

    result = {
        "task_id": task.id,
        "entry_point": task.entry_point,
        "direct": _score_payload(direct_score),
        "spec_guided": _score_payload(spec_score),
        "rescue_lane": "spec_guided",
        "turn_count": len(trace.turns),
        "issue_count": sum(len(turn.issues) for turn in trace.turns),
        "duration_s": round(time.time() - started, 3),
        "llm_calls_made": llm.calls_made,
    }
    if gold_spec is not None:
        oracle = OracleUserProxy(llm, gold_spec_text=gold_spec.model_dump_json(indent=2))
        clarified_trace = run_episode(
            task,
            generator,
            distinguisher,
            oracle,
            max_turns=max_turns,
        )
        if clarified_trace.final_spec is None:
            raise RuntimeError("no clarified spec produced")
        clarified_code = coder.from_spec(clarified_trace.final_spec, task.prompt)
        clarified_score = score_detailed(task, clarified_code)
        gold_code = coder.from_spec(gold_spec, task.prompt)
        gold_score = score_detailed(task, gold_code)
        result["clarified_spec"] = _score_payload(clarified_score)
        result["gold_spec"] = _score_payload(gold_score)
        result["rescue_lane"] = "clarified_spec"
        result["clarified_turn_count"] = len(clarified_trace.turns)
        result["clarified_issue_count"] = sum(
            len(turn.issues) for turn in clarified_trace.turns
        )
        result["clarification_count"] = sum(
            len(turn.user_answers) for turn in clarified_trace.turns
        )
        result["duration_s"] = round(time.time() - started, 3)
        result["llm_calls_made"] = llm.calls_made
    return result


def _run_with_retries(
    task: Task,
    *,
    max_turns: int,
    retries: int,
    generator_prompt: str | None,
    distinguisher_prompt: str | None,
    coder_prompt: str | None,
    gold_spec: Spec | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str | int]] = []
    for attempt in range(retries + 1):
        try:
            result = _run_one(
                task,
                max_turns=max_turns,
                generator_prompt=generator_prompt,
                distinguisher_prompt=distinguisher_prompt,
                coder_prompt=coder_prompt,
                gold_spec=gold_spec,
            )
            result["attempts"] = attempt + 1
            return result
        except Exception as exc:  # noqa: BLE001 - batch runner should keep going.
            errors.append({"attempt": attempt + 1, "error": repr(exc)})
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
    return {
        "task_id": task.id,
        "entry_point": task.entry_point,
        "rescue_lane": "clarified_spec" if gold_spec is not None else "spec_guided",
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
            direct_pass_count = int(result["direct"]["pass_count"])
            direct_total_count = int(result["direct"]["total_count"])
            spec_pass_count = int(result["spec_guided"]["pass_count"])
            spec_total_count = int(result["spec_guided"]["total_count"])
            counters.direct_pass_count += direct_pass_count
            counters.direct_total_count += direct_total_count
            counters.spec_pass_count += spec_pass_count
            counters.spec_total_count += spec_total_count
            clarified_payload = result.get("clarified_spec")
            clarified_ok = (
                bool(clarified_payload["passed_all"])
                if isinstance(clarified_payload, dict)
                else None
            )
            gold_payload = result.get("gold_spec")
            gold_ok = (
                bool(gold_payload["passed_all"]) if isinstance(gold_payload, dict) else None
            )
            rescue_ok = (
                clarified_ok
                if result.get("rescue_lane") == "clarified_spec"
                else spec_ok
            )
            if not direct_ok:
                counters.direct_failed += 1
            if not spec_ok:
                counters.spec_failed += 1
            if clarified_ok is False:
                counters.clarified_failed += 1
            if gold_ok is False:
                counters.gold_failed += 1
            if not direct_ok and rescue_ok:
                counters.rescued += 1
            if not direct_ok and spec_ok:
                counters.generated_rescued += 1
            if not direct_ok and gold_ok:
                counters.gold_rescued += 1
            if direct_ok and rescue_ok is False:
                counters.regressed += 1
            if not direct_ok and rescue_ok is False:
                counters.both_failed += 1
            if spec_pass_count > direct_pass_count:
                counters.partial_improved += 1
            elif spec_pass_count < direct_pass_count:
                counters.partial_regressed += 1
            else:
                counters.partial_unchanged += 1

        direct_pass_rate = (
            counters.direct_pass_count / counters.direct_total_count
            if counters.direct_total_count
            else 0.0
        )
        spec_pass_rate = (
            counters.spec_pass_count / counters.spec_total_count
            if counters.spec_total_count
            else 0.0
        )

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")

        summary = {
            **meta,
            **asdict(counters),
            "direct_pass_rate": direct_pass_rate,
            "spec_pass_rate": spec_pass_rate,
            "spec_minus_direct_pass_count": (
                counters.spec_pass_count - counters.direct_pass_count
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{counters.done}/{meta['task_count']}] "
            f"direct_failed={counters.direct_failed} "
            f"spec_failed={counters.spec_failed} "
            f"clarified_failed={counters.clarified_failed} "
            f"gold_failed={counters.gold_failed} "
            f"rescued={counters.rescued} "
            f"generated_rescued={counters.generated_rescued} "
            f"gold_rescued={counters.gold_rescued} "
            f"regressed={counters.regressed} "
            f"partial_delta={counters.spec_pass_count - counters.direct_pass_count} "
            f"errors={counters.errors} "
            f"task={result.get('task_id')} "
            f"attempts={result.get('attempts', 'ERR')}",
            flush=True,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[
            MBPP_PLUS_DATASET,
            HARD_DATASET,
            HARD_EXTENDED_DATASET,
            HARD_TRAIN_DATASET,
            HARD_DEV_DATASET,
            HARD_TEST_DATASET,
            HARDENED_DATASET,
        ],
        default=MBPP_PLUS_DATASET,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generator-prompt-file", type=Path, default=None)
    parser.add_argument("--distinguisher-prompt-file", type=Path, default=None)
    parser.add_argument("--coder-prompt-file", type=Path, default=None)
    parser.add_argument(
        "--hardened-dir",
        type=Path,
        default=None,
        help="Directory of hardened task JSON files when --dataset hardened.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    tasks, gold_specs = _select_tasks(args)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1
    generator_prompt = _read_prompt_file(args.generator_prompt_file)
    distinguisher_prompt = _read_prompt_file(args.distinguisher_prompt_file)
    coder_prompt = _read_prompt_file(args.coder_prompt_file)

    run_dir = args.run_dir or (
        Path("experiments/runs")
        / f"{args.dataset}-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    runtime = runtime_snapshot()
    meta = {
        "dataset": args.dataset,
        "task_count": len(tasks),
        "workers": args.workers,
        "retries": args.retries,
        "max_turns": args.max_turns,
        **runtime,
        "runtime": runtime,
        "rescue_lane": "clarified_spec" if gold_specs else "spec_guided",
        "gold_spec_lane": bool(gold_specs),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "generator_prompt_file": str(args.generator_prompt_file)
        if args.generator_prompt_file
        else None,
        "distinguisher_prompt_file": str(args.distinguisher_prompt_file)
        if args.distinguisher_prompt_file
        else None,
        "coder_prompt_file": str(args.coder_prompt_file) if args.coder_prompt_file else None,
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
                generator_prompt=generator_prompt,
                distinguisher_prompt=distinguisher_prompt,
                coder_prompt=coder_prompt,
                gold_spec=gold_specs.get(task.id),
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
