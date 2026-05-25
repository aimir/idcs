"""Export evolved G/D prompts from a coevolution run.

Usage:
    python scripts/export_prompts.py experiments/runs/20260525T020955Z
    python scripts/export_prompts.py <run_dir> --epoch 3 --rank 1 --output-dir /tmp/prompts
    python scripts/export_prompts.py <run_dir> --generator-rank 2 --distinguisher-rank 1

The exported ``generator.md`` and ``distinguisher.md`` files can be passed
straight into ``scripts/batch_baseline.py`` for held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = export_prompts(
            args.run_dir,
            output_dir=args.output_dir,
            epoch=args.epoch,
            generator_rank=args.generator_rank or args.rank,
            distinguisher_rank=args.distinguisher_rank or args.rank,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"generator={result['generator_path']}")
    print(f"distinguisher={result['distinguisher_path']}")
    print(f"manifest={result['manifest_path']}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run_dir>/exported_prompts.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to export. Defaults to the latest snapshot epoch.",
    )
    parser.add_argument("--rank", type=int, default=1, help="Rank to export from each role.")
    parser.add_argument(
        "--generator-rank",
        type=int,
        default=None,
        help="Generator rank to export. Defaults to --rank.",
    )
    parser.add_argument(
        "--distinguisher-rank",
        type=int,
        default=None,
        help="Distinguisher rank to export. Defaults to --rank.",
    )
    return parser.parse_args(argv)


def export_prompts(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    epoch: int | None = None,
    rank: int = 1,
    generator_rank: int | None = None,
    distinguisher_rank: int | None = None,
) -> dict[str, str]:
    g_rank = generator_rank or rank
    d_rank = distinguisher_rank or rank
    if rank <= 0 or g_rank <= 0 or d_rank <= 0:
        raise ValueError("ranks must be positive")
    snapshots_dir = run_dir / "prompt_populations"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"missing prompt snapshot directory: {snapshots_dir}")

    epoch_to_export = epoch if epoch is not None else _latest_epoch(snapshots_dir)
    generator = _load_ranked_prompt(snapshots_dir, "generator", epoch_to_export, g_rank)
    distinguisher = _load_ranked_prompt(
        snapshots_dir,
        "distinguisher",
        epoch_to_export,
        d_rank,
    )

    destination = output_dir or (run_dir / "exported_prompts")
    destination.mkdir(parents=True, exist_ok=True)
    generator_path = destination / "generator.md"
    distinguisher_path = destination / "distinguisher.md"
    manifest_path = destination / "manifest.json"

    generator_path.write_text(str(generator["prompt"]).strip() + "\n", encoding="utf-8")
    distinguisher_path.write_text(
        str(distinguisher["prompt"]).strip() + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_dir": str(run_dir),
        "epoch": epoch_to_export,
        "rank": rank,
        "generator_rank": g_rank,
        "distinguisher_rank": d_rank,
        "generator": _manifest_row(generator),
        "distinguisher": _manifest_row(distinguisher),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "generator_path": str(generator_path),
        "distinguisher_path": str(distinguisher_path),
        "manifest_path": str(manifest_path),
    }


def _latest_epoch(snapshots_dir: Path) -> int:
    epochs: list[int] = []
    for path in snapshots_dir.glob("generator_epoch_*.json"):
        try:
            epochs.append(int(path.stem.rsplit("_", maxsplit=1)[-1]))
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError(f"no generator snapshots found in {snapshots_dir}")
    return max(epochs)


def _load_ranked_prompt(
    snapshots_dir: Path,
    role: str,
    epoch: int,
    rank: int,
) -> dict[str, Any]:
    path = snapshots_dir / f"{role}_epoch_{epoch:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {role} snapshot for epoch {epoch}: {path}")
    rows = cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
    for row in rows:
        if row.get("rank") == rank:
            if not row.get("prompt"):
                raise ValueError(f"{role} rank {rank} in {path} has no prompt")
            return row
    raise ValueError(f"rank {rank} not found in {path}")


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_hash": row.get("prompt_hash"),
        "reward": row.get("reward"),
        "avg_benchmark": row.get("avg_benchmark"),
        "anchor": row.get("anchor", False),
        "task_ids": row.get("task_ids", []),
        "frontier_task_ids": row.get("frontier_task_ids", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
