"""Loader for the hand-written seed corpus under ``data/seed_tasks/``.

Each seed file pairs a ``Task`` (with hand-written test assertions) and a
``gold_spec`` representing the user's true intent. The pair lets us drive
the OracleUserProxy during clarification-rate measurement: the oracle
answers from ``gold_spec``, the score comes from ``task.tests``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from idcs.schemas import Spec, Task

SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed_tasks"


@dataclass
class SeedItem:
    """One (task, gold spec) pair from the seed corpus."""

    task: Task
    gold_spec: Spec


def load_seed_corpus(seed_dir: Path | None = None) -> list[SeedItem]:
    """Load every seed item from ``<seed_dir>/*.json`` sorted by filename."""
    directory = seed_dir or SEED_DIR
    items: list[SeedItem] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        items.append(
            SeedItem(
                task=Task.model_validate(data["task"]),
                gold_spec=Spec.model_validate(data["gold_spec"]),
            )
        )
    return items
