"""Run both baselines on MBPP+ and compare pass rates.

Baseline (a): task.prompt → code directly (no spec)
Baseline (b): task.prompt → spec (via orchestrator) → code

Usage:
    uv run python scripts/baseline.py --limit 10
    uv run python scripts/baseline.py --dataset hard --limit 3
    uv run python scripts/baseline.py --tasks Mbpp/2 Mbpp/3
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
import logging
import random
import time
from typing import Any

from idcs.benchmark.scoring import score  # noqa: E402
from idcs.benchmark.tasks import HARD_DATASET, MBPP_PLUS_DATASET, load_benchmark_tasks  # noqa: E402
from idcs.coder import Coder  # noqa: E402
from idcs.distinguisher import Distinguisher  # noqa: E402
from idcs.generator import Generator  # noqa: E402
from idcs.llm import LLM  # noqa: E402
from idcs.orchestrator import run_episode  # noqa: E402
from idcs.schemas import Task  # noqa: E402
from idcs.user_proxy import NullUserProxy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="  %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("baseline")


def run_baseline_a(coder: Coder, task: Task) -> float:
    """Direct: prompt → code."""
    log.info("  (a) coder.from_prompt...")
    t0 = time.time()
    code = coder.from_prompt(task.prompt)
    log.info("  (a) done (%.1fs, %d chars)", time.time() - t0, len(code))
    return score(task, code)


def run_baseline_b(
    coder: Coder,
    generator: Generator,
    distinguisher: Distinguisher,
    task: Task,
    max_turns: int = 3,
) -> float:
    """Spec-guided: prompt → spec → code."""
    log.info("  (b) orchestrator (max_turns=%d)...", max_turns)
    t0 = time.time()
    trace = run_episode(task, generator, distinguisher, NullUserProxy(), max_turns=max_turns)
    log.info(
        "  (b) spec done (%.1fs, %d turns)",
        time.time() - t0,
        len(trace.turns),
    )
    if trace.final_spec is None:
        return 0.0
    log.info("  (b) coder.from_spec...")
    t1 = time.time()
    code = coder.from_spec(trace.final_spec, task.prompt)
    log.info("  (b) done (%.1fs, %d chars)", time.time() - t1, len(code))
    return score(task, code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[MBPP_PLUS_DATASET, HARD_DATASET],
        default=MBPP_PLUS_DATASET,
        help="benchmark slice to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="max tasks to run")
    parser.add_argument("--offset", type=int, default=0, help="skip first N tasks")
    parser.add_argument("--sample", type=int, default=None, help="random sample of N tasks")
    parser.add_argument("--min-tests", type=int, default=0, help="only tasks with >= N assertions")
    parser.add_argument("--tasks", nargs="*", default=None, help="specific task IDs")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tasks = load_benchmark_tasks(args.dataset)
    total_available = len(tasks)
    if args.tasks:
        tasks = [t for t in tasks if t.id in args.tasks]
    if args.min_tests:
        tasks = [t for t in tasks if len(t.tests) >= args.min_tests]
    if args.offset:
        tasks = tasks[args.offset:]
    if args.sample:
        rng = random.Random(args.seed)
        tasks = rng.sample(tasks, min(args.sample, len(tasks)))
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    llm = LLM()

    print(f"\n{'='*50}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {llm.model}")
    print(f"Tasks:      {len(tasks)} (of {total_available} available)")
    if args.offset:
        print(f"Offset:     {args.offset}")
    if args.min_tests:
        print(f"Min tests:  {args.min_tests}")
    if args.sample:
        print(f"Sample:     {args.sample} (seed={args.seed})")
    print(f"Max turns:  {args.max_turns}")
    print(f"Task IDs:   {tasks[0].id} .. {tasks[-1].id}")
    print(f"{'='*50}\n")

    coder = Coder(llm)
    generator = Generator(llm)
    distinguisher = Distinguisher(llm)

    results: list[dict[str, Any]] = []
    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.id}")
        t_task = time.time()
        score_a = run_baseline_a(coder, task)
        score_b = run_baseline_b(coder, generator, distinguisher, task, args.max_turns)
        tag = "WIN" if score_b > score_a else ("LOSE" if score_b < score_a else "TIE")
        elapsed = time.time() - t_task
        results.append({"task_id": task.id, "a": score_a, "b": score_b})
        print(f"  => (a)={score_a:.2f}  (b)={score_b:.2f}  {tag}  [{elapsed:.1f}s]\n")

    total_time = time.time() - t_start
    avg_a = sum(r["a"] for r in results) / len(results)
    avg_b = sum(r["b"] for r in results) / len(results)
    wins = sum(1 for r in results if r["b"] > r["a"])
    losses = sum(1 for r in results if r["b"] < r["a"])
    ties = sum(1 for r in results if r["b"] == r["a"])

    print(f"{'='*50}")
    print(f"Aggregate: (a)={avg_a:.3f}  (b)={avg_b:.3f}  (plus-input tests only)")
    print(f"Record: {wins}W / {losses}L / {ties}T")
    print(f"Time: {total_time:.0f}s total, {total_time/len(results):.1f}s/task")
    if avg_b >= avg_a:
        print("Phase 2 EXIT CRITERION MET: (b) does not regress vs (a)")
    else:
        print("(b) regresses vs (a) — iterate on spec/coder prompts")
    return 0 if avg_b >= avg_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
