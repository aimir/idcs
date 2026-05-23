"""Telemetry helpers for coevolution runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idcs.schemas import Trace


def create_run_dir(root: Path | None = None) -> Path:
    base = root or (Path(__file__).resolve().parents[2] / "experiments" / "runs")
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / timestamp
    if not run_dir.exists():
        run_dir.mkdir()
        return run_dir
    counter = 1
    while True:
        candidate = base / f"{timestamp}-{counter}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        counter += 1


def write_trace(run_dir: Path, trace: Trace) -> None:
    _append_jsonl(run_dir / "traces.jsonl", trace.model_dump())


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    _append_jsonl(run_dir / "metrics.jsonl", metrics)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
