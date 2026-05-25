"""Launch multiple coevolution runs in parallel.

This is a convenience wrapper around ``scripts/train.py`` for fast prompt
search. Each run gets a different seed, the same pinned G/D/coder model, and
the same pinned mutator model. Stdout/stderr are written under
``experiments/parallel/<timestamp>/``.

Example:
    python scripts/parallel_train.py --benchmark hard --seeds 101 102 103 \
        --jobs 3 --model gpt-5.5 --mutator-model gpt-5.5
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR_RE = re.compile(r"Run dir:\s*(.+)")
BEST_G_RE = re.compile(r"Best generator reward:\s*([-0-9.]+)")
BEST_D_RE = re.compile(r"Best distinguisher reward:\s*([-0-9.]+)")


@dataclass(frozen=True)
class CompletedRun:
    seed: int
    returncode: int
    seconds: float
    stdout_path: Path
    stderr_path: Path
    run_dir: str | None
    best_generator: str | None
    best_distinguisher: str | None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    jobs = args.jobs or len(args.seeds)
    log_root = args.output_dir or _default_log_root()
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(args.seeds)} runs with {jobs} workers.")
    print(f"model={args.model} mutator={args.mutator_model or args.model}")
    print(f"logs={log_root}")

    results: list[CompletedRun] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one, args, seed, log_root): seed
            for seed in args.seeds
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
            print(
                f"seed={result.seed} {status} {result.seconds:.1f}s "
                f"G={result.best_generator or '?'} D={result.best_distinguisher or '?'} "
                f"run={result.run_dir or '-'}"
            )

    print("\nsummary:")
    for result in sorted(results, key=lambda item: item.seed):
        print(
            f"{result.seed}: exit={result.returncode} "
            f"G={result.best_generator or '?'} D={result.best_distinguisher or '?'} "
            f"stdout={result.stdout_path}"
        )
    return 0 if all(result.returncode == 0 for result in results) else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["mbpp", "hard", "seed"], default="hard")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026052501, 2026052502])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--mutator-model", type=str, default=None)
    parser.add_argument("--backend", type=str, default="codex")
    parser.add_argument("--service-tier", type=str, default="fast")
    parser.add_argument("--reasoning-effort", type=str, default="none")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="specific task IDs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=3)
    parser.add_argument("--elite-size", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--task-sample", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--generator-prompt-file", type=Path, default=None)
    parser.add_argument("--distinguisher-prompt-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _run_one(args: argparse.Namespace, seed: int, log_root: Path) -> CompletedRun:
    command = _train_command(args, seed)
    env = _run_env(args)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    seconds = time.monotonic() - start
    stdout_path = log_root / f"seed-{seed}.stdout.log"
    stderr_path = log_root / f"seed-{seed}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    combined = f"{completed.stdout}\n{completed.stderr}"
    return CompletedRun(
        seed=seed,
        returncode=completed.returncode,
        seconds=seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        run_dir=_first_match(RUN_DIR_RE, combined),
        best_generator=_first_match(BEST_G_RE, combined),
        best_distinguisher=_first_match(BEST_D_RE, combined),
    )


def _train_command(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--benchmark",
        args.benchmark,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--pop-size",
        str(args.pop_size),
        "--elite-size",
        str(args.elite_size),
        "--max-turns",
        str(args.max_turns),
        "--model",
        args.model,
        "--mutator-model",
        args.mutator_model or args.model,
    ]
    _append_optional(command, "--limit", args.limit)
    _append_optional(command, "--offset", args.offset)
    _append_optional(command, "--sample", args.sample)
    if args.tasks:
        command.append("--tasks")
        command.extend(args.tasks)
    _append_optional(command, "--task-sample", args.task_sample)
    _append_optional(command, "--val-fraction", args.val_fraction)
    _append_optional(command, "--max-llm-calls", args.max_llm_calls)
    _append_optional(command, "--generator-prompt-file", args.generator_prompt_file)
    _append_optional(command, "--distinguisher-prompt-file", args.distinguisher_prompt_file)
    return command


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _run_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["IDCS_BACKEND"] = args.backend
    env["IDCS_CODEX_MODEL"] = args.model
    env["IDCS_MUTATOR_MODEL"] = args.mutator_model or args.model
    env["IDCS_CODEX_TIMEOUT_S"] = str(args.timeout_s)
    if args.service_tier:
        env["IDCS_CODEX_SERVICE_TIER"] = args.service_tier
    if args.reasoning_effort:
        env["IDCS_CODEX_REASONING_EFFORT"] = args.reasoning_effort
    return env


def _default_log_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "parallel" / stamp


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


if __name__ == "__main__":
    raise SystemExit(main())
