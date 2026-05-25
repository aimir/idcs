"""Run coevolution (G + D) prompt optimization."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idcs._prompts import load_prompt  # noqa: E402
from idcs.benchmark.tasks import load_mbpp_plus  # noqa: E402
from idcs.llm import LLM, BudgetExceededError  # noqa: E402
from idcs.optimizer.coevolve import CoevolveConfig, coevolve  # noqa: E402
from idcs.schemas import Task  # noqa: E402
from idcs.seed_corpus import load_seed_corpus  # noqa: E402
from idcs.user_proxy import NullUserProxy, OracleUserProxy, UserProxy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")
# httpx logs every HTTP request at INFO; our own progress logs supersede that
# and the POST-200 lines just create noise. Bump it to WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["mbpp", "seed"], default="mbpp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--task-sample", type=int, default=None)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of tasks for monitor-only val eval per epoch (e.g. 0.2).",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help=(
            "Hard ceiling on total LLM API calls (main + mutator). "
            "Once hit, the next call raises BudgetExceededError and the run exits."
        ),
    )
    args = parser.parse_args()

    llm = LLM(max_calls=args.max_llm_calls)
    # Optional cheaper model for the mutator. Defaults to the main LLM if
    # IDCS_MUTATOR_MODEL is unset.
    mutator_model = os.environ.get("IDCS_MUTATOR_MODEL")
    mutator_llm = LLM(model=mutator_model) if mutator_model else None
    if mutator_llm is not None:
        print(f"Using {mutator_llm.model} for prompt mutations (main: {llm.model}).")
    generator_prompt = load_prompt("generator_v0")
    distinguisher_prompt = load_prompt("distinguisher_v0")
    tasks: list[Task]
    user_factory: Callable[[Task], UserProxy]

    if args.benchmark == "seed":
        items = load_seed_corpus()
        tasks = [item.task for item in items]
        gold_map = {item.task.id: item.gold_spec for item in items}

        def seed_user_factory(task: Task) -> UserProxy:
            gold = gold_map[task.id]
            return OracleUserProxy(llm, gold_spec_text=gold.model_dump_json(indent=2))

        user_factory = seed_user_factory

    else:
        tasks = load_mbpp_plus()

        def null_user_factory(task: Task) -> UserProxy:
            return NullUserProxy()

        user_factory = null_user_factory

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

    val_tasks: list[Task] = []
    if 0.0 < args.val_fraction < 1.0 and len(tasks) >= 5:
        rng_split = random.Random(args.seed)
        shuffled = list(tasks)
        rng_split.shuffle(shuffled)
        split_idx = max(1, int(round(len(shuffled) * args.val_fraction)))
        val_tasks = shuffled[:split_idx]
        tasks = shuffled[split_idx:]
        print(f"Held out {len(val_tasks)} val tasks; training on {len(tasks)}.")

    config = CoevolveConfig(
        population_size=args.pop_size,
        elite_size=args.elite_size,
        epochs=args.epochs,
        max_turns=args.max_turns,
        task_sample_size=args.task_sample,
        seed=args.seed,
        telemetry=not args.no_telemetry,
    )

    try:
        result = coevolve(
            tasks,
            llm,
            generator_prompt=generator_prompt,
            distinguisher_prompt=distinguisher_prompt,
            user_factory=user_factory,
            config=config,
            val_tasks=val_tasks or None,
            mutator_llm=mutator_llm,
        )
    except BudgetExceededError as e:
        print(f"\nBUDGET EXHAUSTED: {e}", file=sys.stderr)
        print(f"LLM calls used: {llm.calls_made}", file=sys.stderr)
        return 2

    print(f"Run dir: {result.run_dir}" if result.run_dir else "Telemetry disabled")
    print(f"Best generator reward: {result.generator.best().reward:.4f}")
    print(f"Best distinguisher reward: {result.distinguisher.best().reward:.4f}")
    print(f"LLM calls used: {llm.calls_made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
