"""Run coevolution (G + D) prompt optimization."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idcs._prompts import load_prompt  # noqa: E402
from idcs.benchmark.tasks import load_mbpp_plus  # noqa: E402
from idcs.llm import LLM  # noqa: E402
from idcs.optimizer.coevolve import CoevolveConfig, coevolve  # noqa: E402
from idcs.seed_corpus import load_seed_corpus  # noqa: E402
from idcs.user_proxy import NullUserProxy, OracleUserProxy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")


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
    args = parser.parse_args()

    llm = LLM()
    generator_prompt = load_prompt("generator_v0")
    distinguisher_prompt = load_prompt("distinguisher_v0")

    if args.benchmark == "seed":
        items = load_seed_corpus()
        tasks = [item.task for item in items]
        gold_map = {item.task.id: item.gold_spec for item in items}

        def user_factory(task):
            gold = gold_map[task.id]
            return OracleUserProxy(llm, gold_spec_text=gold.model_dump_json(indent=2))

    else:
        tasks = load_mbpp_plus()

        def user_factory(task):
            return NullUserProxy()

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

    config = CoevolveConfig(
        population_size=args.pop_size,
        elite_size=args.elite_size,
        epochs=args.epochs,
        max_turns=args.max_turns,
        task_sample_size=args.task_sample,
        seed=args.seed,
        telemetry=not args.no_telemetry,
    )

    result = coevolve(
        tasks,
        llm,
        generator_prompt=generator_prompt,
        distinguisher_prompt=distinguisher_prompt,
        user_factory=user_factory,
        config=config,
    )

    print(f"Run dir: {result.run_dir}" if result.run_dir else "Telemetry disabled")
    print(f"Best generator reward: {result.generator.best().reward:.4f}")
    print(f"Best distinguisher reward: {result.distinguisher.best().reward:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
