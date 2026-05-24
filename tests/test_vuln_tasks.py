"""Validate vulnerability benchmark tasks load correctly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idcs.schemas import Task

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed_tasks"


def _is_vuln_task(path: Path) -> bool:
    """Vuln tasks are numbered 09 and above."""
    head = path.stem.split("-", 1)[0]
    return head.isdigit() and int(head) >= 9


VULN_TASKS = sorted(SEED_DIR.glob("*-*.json"))
VULN_ONLY = [p for p in VULN_TASKS if _is_vuln_task(p)]


@pytest.mark.parametrize("path", VULN_ONLY, ids=lambda p: p.stem)
def test_vuln_task_loads(path: Path) -> None:
    """Each vulnerability task JSON has valid task, gold_spec, and known_weakness."""
    data = json.loads(path.read_text())

    # Task parses correctly
    task = Task(**data["task"])
    assert task.id.startswith("vuln/")
    assert len(task.tests) >= 3

    # Gold spec is present and has postconditions
    gold = data["gold_spec"]
    assert gold["goal"]
    assert gold["postconditions"], f"{path.stem}: gold_spec missing postconditions"

    # Known weakness is present with required fields
    weakness = data["known_weakness"]
    assert weakness["type"] in {
        "underconstraint", "implicit_assumption", "ambiguity",
        "contradiction", "gap", "over_constraint",
    }, f"Unknown weakness type: {weakness['type']}"
    assert weakness["description"]
    assert weakness["naive_spec_gap"]


def test_vuln_tasks_exist() -> None:
    """At least 4 vulnerability tasks should be present."""
    assert len(VULN_ONLY) >= 4, f"Only {len(VULN_ONLY)} vuln tasks found"
