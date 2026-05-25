"""Telemetry helpers for coevolution runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idcs.schemas import Trace


def create_run_dir(root: Path | None = None) -> Path:
    base = root or (Path(__file__).resolve().parents[2] / "experiments" / "runs")
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Use mkdir(exist_ok=False) so directory creation is atomic; two runs
    # started in the same second from concurrent processes race-safely fall
    # through to the suffix loop instead of one of them throwing.
    candidate = base / timestamp
    counter = 0
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            counter += 1
            candidate = base / f"{timestamp}-{counter}"


def write_trace(run_dir: Path, trace: Trace) -> None:
    _append_jsonl(run_dir / "traces.jsonl", trace.model_dump())


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    _append_jsonl(run_dir / "metrics.jsonl", metrics)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
