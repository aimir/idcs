"""Browse a coevolution run directory.

Reads ``metrics.jsonl`` (per-task and per-epoch metrics) and
``traces.jsonl`` (full episode records) plus the ``config.json`` snapshot
that lives alongside them.

(Named ``inspect_run.py`` — not ``inspect.py`` — to avoid shadowing the
stdlib ``inspect`` module when Python adds ``scripts/`` to ``sys.path``.)

Usage:
    python scripts/inspect_run.py <run_dir>                    # summary
    python scripts/inspect_run.py <run_dir> --epoch 3           # rows from epoch 3
    python scripts/inspect_run.py <run_dir> --task Mbpp/42      # rows for one task
    python scripts/inspect_run.py <run_dir> --trace Mbpp/42 \\
        --prompt-hash abc123ef                                  # one full trace
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _print_config(run_dir: Path) -> None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        print("(no config.json — run predates the snapshot, or telemetry was off)")
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(f"model:        {cfg.get('model')}")
    print(f"mutator:      {cfg.get('mutator_model') or cfg.get('model')}")
    weights = cfg.get("weights", {})
    print(f"weights:      α={weights.get('alpha')}  β={weights.get('beta')}  "
          f"γ={weights.get('gamma')}  δ={weights.get('delta')}  ε={weights.get('epsilon')}  "
          f"max_t2={weights.get('max_type2_per_episode')}")
    config = cfg.get("config", {})
    print(f"pop/elite:    {config.get('population_size')}/{config.get('elite_size')}  "
          f"epochs={config.get('epochs')}  max_turns={config.get('max_turns')}  "
          f"task_sample={config.get('task_sample_size')}")
    train_ids = cfg.get("train_task_ids", [])
    val_ids = cfg.get("val_task_ids", [])
    print(f"tasks:        {len(train_ids)} train, {len(val_ids)} val")
    baselines = cfg.get("baselines", {})
    if baselines:
        bvals = list(baselines.values())
        print(f"no-spec baseline: avg={mean(bvals):.3f} over {len(bvals)} tasks")


def _print_summary(metrics: list[dict[str, Any]]) -> None:
    """Per-epoch best/avg reward for each role, plus val numbers."""
    # Per-epoch aggregate rows have keys: epoch, role, best_reward, avg_reward
    # Val rows have: epoch, split="val", val_avg_*
    by_epoch_role: dict[tuple[int, str], dict[str, Any]] = {}
    val_by_epoch: dict[int, dict[str, Any]] = {}
    for row in metrics:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        if row.get("split") == "val":
            val_by_epoch[epoch] = row
            continue
        if "best_reward" in row and "role" in row:
            by_epoch_role[(epoch, row["role"])] = row

    if not by_epoch_role:
        print("(no per-epoch aggregate rows in metrics.jsonl)")
        return

    epochs = sorted({e for e, _ in by_epoch_role})
    header = (
        f"{'epoch':>6}  "
        f"{'G best':>8}  {'G avg':>8}  "
        f"{'D best':>8}  {'D avg':>8}  "
        f"{'val benchmark':>14}  {'val rG':>8}  {'val rD':>8}"
    )
    print(header)
    print("-" * len(header))
    for epoch in epochs:
        g = by_epoch_role.get((epoch, "generator"), {})
        d = by_epoch_role.get((epoch, "distinguisher"), {})
        v = val_by_epoch.get(epoch, {})
        print(
            f"{epoch:>6}  "
            f"{g.get('best_reward', float('nan')):>8.3f}  "
            f"{g.get('avg_reward', float('nan')):>8.3f}  "
            f"{d.get('best_reward', float('nan')):>8.3f}  "
            f"{d.get('avg_reward', float('nan')):>8.3f}  "
            f"{v.get('val_avg_benchmark', float('nan')):>14.3f}  "
            f"{v.get('val_avg_r_generator', float('nan')):>8.3f}  "
            f"{v.get('val_avg_r_distinguisher', float('nan')):>8.3f}"
        )


def _print_per_task_summary(metrics: list[dict[str, Any]]) -> None:
    """For each task, aggregate reward stats across all candidate evaluations."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in metrics:
        if "task_id" in row and "reward" in row:
            by_task[row["task_id"]].append(row["reward"])
    if not by_task:
        return
    print("\nper-task reward (averaged across all candidate evals):")
    print(f"{'task':<40}  {'n':>4}  {'mean':>8}  {'min':>8}  {'max':>8}")
    for task_id in sorted(by_task):
        rewards = by_task[task_id]
        print(
            f"{task_id:<40}  {len(rewards):>4}  "
            f"{mean(rewards):>8.3f}  {min(rewards):>8.3f}  {max(rewards):>8.3f}"
        )


def _print_llm_telemetry(metrics: list[dict[str, Any]]) -> None:
    counts = [
        row["llm_structured_fallback_count"]
        for row in metrics
        if isinstance(row.get("llm_structured_fallback_count"), int | float)
    ]
    if counts:
        print(f"\nstructured-output fallbacks: {max(counts):.0f}")


def _filter_metrics(
    metrics: list[dict[str, Any]],
    *,
    epoch: int | None,
    task: str | None,
    role: str | None,
) -> list[dict[str, Any]]:
    rows = metrics
    if epoch is not None:
        rows = [r for r in rows if r.get("epoch") == epoch]
    if task is not None:
        rows = [r for r in rows if r.get("task_id") == task]
    if role is not None:
        rows = [r for r in rows if r.get("role") == role]
    return rows


def _filter_traces(
    traces: list[dict[str, Any]],
    *,
    task: str,
    prompt_hash: str | None,
) -> list[dict[str, Any]]:
    rows = [t for t in traces if t.get("task_id") == task]
    if prompt_hash is not None:
        rows = [
            t for t in rows
            if t.get("generator_prompt_hash") == prompt_hash
            or t.get("distinguisher_prompt_hash") == prompt_hash
        ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--epoch", type=int, help="filter rows to this epoch")
    parser.add_argument("--task", type=str, help="filter rows to this task_id")
    parser.add_argument(
        "--role", choices=["generator", "distinguisher"], help="filter to one role"
    )
    parser.add_argument(
        "--trace",
        type=str,
        metavar="TASK_ID",
        help="dump the full Trace JSON for this task (use with --prompt-hash to disambiguate)",
    )
    parser.add_argument("--prompt-hash", type=str, help="filter traces by prompt hash")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Not a directory: {args.run_dir}", file=sys.stderr)
        return 1

    metrics = _load_jsonl(args.run_dir / "metrics.jsonl")
    traces = _load_jsonl(args.run_dir / "traces.jsonl")

    if args.trace:
        matched = _filter_traces(traces, task=args.trace, prompt_hash=args.prompt_hash)
        if not matched:
            print(f"No trace matches task={args.trace} prompt_hash={args.prompt_hash}")
            return 1
        for trace in matched:
            print(json.dumps(trace, indent=2, ensure_ascii=False))
            print()
        return 0

    if args.epoch is not None or args.task is not None or args.role is not None:
        filtered = _filter_metrics(metrics, epoch=args.epoch, task=args.task, role=args.role)
        for row in filtered:
            print(json.dumps(row, ensure_ascii=False))
        return 0

    # Default: configuration summary + per-epoch aggregate + per-task aggregate.
    print(f"=== {args.run_dir} ===\n")
    _print_config(args.run_dir)
    print()
    _print_summary(metrics)
    _print_llm_telemetry(metrics)
    _print_per_task_summary(metrics)
    print(f"\n{len(metrics)} metric rows, {len(traces)} traces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
