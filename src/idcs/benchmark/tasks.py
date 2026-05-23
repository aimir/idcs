"""Task-loading adapter for the EvalPlus benchmark library.

EvalPlus's MBPP+ shape is normalized into our ``Task`` / ``Test`` schema so
the rest of the pipeline (G, D, orchestrator) sees a single Task type
regardless of source. Scoring will route back through EvalPlus in a
follow-up — see ``benchmark/scoring.py`` (TODO Phase 2).

EvalPlus is imported lazily so that importing this module does not drag in
its full dependency tree (numpy, datasets, transformers, ...).
"""

from __future__ import annotations

from typing import Any

from idcs.schemas import Task, Test


def load_mbpp_plus() -> list[Task]:
    """Return every MBPP+ task, normalized to our ``Task`` schema."""
    from evalplus.data import get_mbpp_plus

    return [_to_task(task_id, item) for task_id, item in get_mbpp_plus().items()]


def _to_task(task_id: str, item: dict[str, Any]) -> Task:
    prompt = str(item.get("prompt", ""))
    assertion_block = str(item.get("assertion", ""))
    tests = [
        Test(id=f"{task_id}/t{i}", code=line)
        for i, line in enumerate(_split_assertions(assertion_block))
    ]
    return Task(id=task_id, prompt=prompt, tests=tests)


def _split_assertions(text: str) -> list[str]:
    """Pull out each ``assert ...`` line from a multi-assert block."""
    return [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()).startswith("assert")
    ]
