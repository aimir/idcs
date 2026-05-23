"""Run both baselines on MBPP+ and compare pass rates.

Baseline (a): task.prompt → code directly (no spec)
Baseline (b): task.prompt → spec (via orchestrator) → code

Usage:
    uv run python scripts/baseline.py --limit 10
    uv run python scripts/baseline.py --tasks Mbpp/2 Mbpp/3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idcs.benchmark.scoring import score
from idcs.benchmark.tasks import load_mbpp_plus
from idcs.coder import Coder
from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.llm import LLM
from idcs.orchestrator import run_episode
from idcs.schemas import Task
from idcs.user_proxy import NullUserProxy


def run_baseline_a(coder: Coder, task: Task) -> float:
    """Direct: prompt → code."""
    code = coder.from_prompt(task.prompt)
    return score(task, code)


def run_baseline_b(
    coder: Coder,
    generator: Generator,
    distinguisher: Distinguisher,
    task: Task,
    max_turns: int = 3,
) -> float:
    """Spec-guided: prompt → spec → code."""
    trace = run_episode(task, generator, distinguisher, NullUserProxy(), max_turns=max_turns)
    if trace.final_spec is None:
        return 0.0
    code = coder.from_spec(trace.final_spec, task.prompt)
    return score(task, code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    tasks = load_mbpp_plus()
    if args.tasks:
        tasks = [t for t in tasks if t.id in args.tasks]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Running baselines on {len(tasks)} tasks...\n")

    llm = LLM()
    coder = Coder(llm)
    generator = Generator(llm)
    distinguisher = Distinguisher(llm)

    results: list[dict] = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.id}...", end=" ", flush=True)
        score_a = run_baseline_a(coder, task)
        score_b = run_baseline_b(coder, generator, distinguisher, task, args.max_turns)
        tag = "WIN" if score_b > score_a else ("LOSE" if score_b < score_a else "TIE")
        results.append({"task_id": task.id, "a": score_a, "b": score_b})
        print(f"(a)={score_a:.2f}  (b)={score_b:.2f}  {tag}")

    avg_a = sum(r["a"] for r in results) / len(results)
    avg_b = sum(r["b"] for r in results) / len(results)
    wins = sum(1 for r in results if r["b"] > r["a"])
    losses = sum(1 for r in results if r["b"] < r["a"])
    ties = sum(1 for r in results if r["b"] == r["a"])

    print(f"\n{'='*50}")
    print(f"Aggregate: (a)={avg_a:.3f}  (b)={avg_b:.3f}")
    print(f"Record: {wins}W / {losses}L / {ties}T")
    if avg_b > avg_a:
        print("Phase 2 EXIT CRITERION MET: (b) beats (a)")
    else:
        print("(b) does NOT beat (a) — iterate on spec/coder prompts")
    return 0 if avg_b > avg_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
