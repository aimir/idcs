"""Run the G → D → user loop on each seed task and print the trace.

Requires ``OPENROUTER_API_KEY`` in the environment. Optional:
``IDCS_MODEL`` to override the default model (e.g. ``openai/gpt-4o``,
``google/gemini-2.5-pro``).

Usage:
    python scripts/cold_start.py                # all seed tasks
    python scripts/cold_start.py 01 03          # only matching task ids
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.llm import LLM
from idcs.orchestrator import run_episode
from idcs.schemas import Spec, Task
from idcs.user_proxy import OracleUserProxy

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed_tasks"


def main(argv: list[str]) -> int:
    files = sorted(SEED_DIR.glob("*.json"))
    if argv:
        files = [f for f in files if any(a in f.stem for a in argv)]
    if not files:
        print("No seed tasks matched.", file=sys.stderr)
        return 1

    llm = LLM()
    generator = Generator(llm)
    distinguisher = Distinguisher(llm)

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        task = Task.model_validate(data["task"])
        gold_spec = Spec.model_validate(data["gold_spec"])
        user = OracleUserProxy(llm, gold_spec_text=gold_spec.model_dump_json(indent=2))

        header = f"{task.id}: {task.prompt}"
        print(f"\n{'=' * 70}\n{header}\n{'=' * 70}")
        trace = run_episode(task, generator, distinguisher, user)
        print(json.dumps(trace.model_dump(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
