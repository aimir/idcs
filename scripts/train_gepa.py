"""Run GEPA prompt optimization for the IDCS pipeline.

This is an isolated experiment lane. It does not replace ``scripts/train.py``.

Example:
    IDCS_BACKEND=codex IDCS_CODEX_MODEL=gpt-5.4-mini \\
      uv run --no-project --with '.[dev]' --with gepa \\
      python scripts/train_gepa.py --dataset hard --limit 3 --max-metric-calls 20
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
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from idcs._prompts import load_prompt
from idcs.benchmark.tasks import HARD_DATASET, MBPP_PLUS_DATASET, load_benchmark_tasks
from idcs.llm import LLM, BudgetExceededError
from idcs.optimizer.gepa_adapter import (
    IDCSGepaAdapter,
    compute_direct_baselines,
    seed_candidate,
)
from idcs.schemas import Task
from idcs.seed_corpus import load_seed_corpus
from idcs.user_proxy import NullUserProxy, OracleUserProxy, UserProxy

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        import gepa
    except ImportError:
        print(
            "GEPA is not installed. Re-run with: "
            "uv run --no-project --with '.[dev]' --with gepa python scripts/train_gepa.py ...",
            file=sys.stderr,
        )
        return 2

    llm = LLM(model=args.model, max_calls=args.max_llm_calls)
    generator_prompt = _read_prompt(args.generator_prompt_file, "generator_v0")
    distinguisher_prompt = _read_prompt(args.distinguisher_prompt_file, "distinguisher_v0")
    coder_prompt = (
        args.coder_prompt_file.read_text(encoding="utf-8")
        if args.coder_prompt_file is not None
        else None
    )

    tasks, user_factory = _load_tasks(args, llm)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    val_tasks: list[Task] = []
    if 0.0 < args.val_fraction < 1.0 and len(tasks) >= 5:
        rng = random.Random(args.seed)
        shuffled = list(tasks)
        rng.shuffle(shuffled)
        split_idx = max(1, int(round(len(shuffled) * args.val_fraction)))
        val_tasks = shuffled[:split_idx]
        tasks = shuffled[split_idx:]
        print(f"Held out {len(val_tasks)} val tasks; training on {len(tasks)}.")

    baseline_tasks = list({task.id: task for task in [*tasks, *val_tasks]}.values())
    baselines: dict[str, float] = {}
    if not args.skip_baselines:
        print(f"Computing {len(baseline_tasks)} direct baselines...")
        baselines = compute_direct_baselines(
            baseline_tasks,
            llm,
            coder_prompt=coder_prompt,
        )

    run_dir = args.run_dir or (
        Path("experiments/runs")
        / f"gepa-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_invocation(
        run_dir,
        args=args,
        tasks=tasks,
        val_tasks=val_tasks,
        baselines=baselines,
        model=llm.model,
    )

    adapter = IDCSGepaAdapter(
        llm=llm,
        generator_prompt=generator_prompt,
        distinguisher_prompt=distinguisher_prompt,
        coder_prompt=coder_prompt,
        user_factory=user_factory,
        baseline_scores=baselines,
        max_turns=args.max_turns,
    )

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate(
                generator_prompt=generator_prompt,
                distinguisher_prompt=distinguisher_prompt,
                coder_prompt=coder_prompt if args.optimize_coder else None,
            ),
            trainset=tasks,
            valset=val_tasks or tasks,
            adapter=adapter,
            reflection_lm=args.reflection_model,
            candidate_selection_strategy=args.candidate_selection,
            frontier_type=args.frontier_type,
            module_selector=args.module_selector,
            max_metric_calls=args.max_metric_calls,
            run_dir=str(run_dir / "gepa"),
            track_best_outputs=True,
            display_progress_bar=args.progress,
            cache_evaluation=args.cache_evaluation,
            seed=args.seed,
            raise_on_exception=False,
        )
    except BudgetExceededError as exc:
        print(f"\nBUDGET EXHAUSTED: {exc}", file=sys.stderr)
        print(f"LLM calls used: {llm.calls_made}", file=sys.stderr)
        return 3

    summary = {
        "best_idx": result.best_idx,
        "best_candidate": result.best_candidate,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "val_aggregate_scores": result.val_aggregate_scores,
        "llm_calls_used": llm.calls_made,
        "gepa_run_dir": result.run_dir,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Run dir: {run_dir}")
    print(f"Best candidate index: {result.best_idx}")
    print(f"Candidates: {result.num_candidates}")
    print(f"Metric calls: {result.total_metric_calls}")
    print(f"LLM calls used: {llm.calls_made}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[MBPP_PLUS_DATASET, HARD_DATASET, "seed"],
        default=HARD_DATASET,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="specific task IDs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the IDCS G/D/coder model.",
    )
    parser.add_argument("--reflection-model", type=str, default=None, help="GEPA reflection LM.")
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=25)
    parser.add_argument(
        "--candidate-selection",
        choices=["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
        default="pareto",
    )
    parser.add_argument(
        "--frontier-type",
        choices=["instance", "objective", "hybrid", "cartesian"],
        default="instance",
    )
    parser.add_argument("--module-selector", default="round_robin")
    parser.add_argument("--cache-evaluation", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--optimize-coder", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generator-prompt-file", type=Path, default=None)
    parser.add_argument("--distinguisher-prompt-file", type=Path, default=None)
    parser.add_argument("--coder-prompt-file", type=Path, default=None)
    return parser.parse_args(argv)


def _read_prompt(path: Path | None, default_name: str) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return load_prompt(default_name)


def _load_tasks(
    args: argparse.Namespace,
    llm: LLM,
) -> tuple[list[Task], Callable[[Task], UserProxy]]:
    if args.dataset == "seed":
        seed_items = load_seed_corpus()
        tasks = [item.task for item in seed_items]
        gold_map = {item.task.id: item.gold_spec for item in seed_items}

        def seed_user_factory(task: Task) -> UserProxy:
            return OracleUserProxy(
                llm,
                gold_spec_text=gold_map[task.id].model_dump_json(indent=2),
            )

        user_factory = seed_user_factory
    else:
        tasks = load_benchmark_tasks(args.dataset)

        def null_user_factory(task: Task) -> UserProxy:
            del task
            return NullUserProxy()

        user_factory = null_user_factory

    if args.tasks:
        task_ids = set(args.tasks)
        tasks = [task for task in tasks if task.id in task_ids]
        missing = sorted(task_ids - {task.id for task in tasks})
        if missing:
            raise SystemExit(f"Task IDs not found: {', '.join(missing)}")
    if args.offset:
        tasks = tasks[args.offset :]
    if args.sample:
        rng = random.Random(args.seed)
        tasks = rng.sample(tasks, min(args.sample, len(tasks)))
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks, user_factory


def _write_invocation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    tasks: Sequence[Task],
    val_tasks: Sequence[Task],
    baselines: Mapping[str, float],
    model: str,
) -> None:
    payload: dict[str, Any] = {
        "args": vars(args),
        "model": model,
        "train_task_ids": [task.id for task in tasks],
        "val_task_ids": [task.id for task in val_tasks],
        "baselines": dict(baselines),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
