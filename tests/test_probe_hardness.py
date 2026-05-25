from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_probe_hardness():
    script = Path(__file__).resolve().parents[1] / "scripts" / "probe_hardness.py"
    spec = importlib.util.spec_from_file_location("probe_hardness", script)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load probe_hardness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_score_prioritizes_solved() -> None:
    probe = _load_probe_hardness()

    assert probe.classify_score(base_pass_rate=1.0, plus_pass_rate=1.0) == "solved"


def test_classify_score_marks_base_ok_plus_fail_as_high_value() -> None:
    probe = _load_probe_hardness()

    assert (
        probe.classify_score(base_pass_rate=1.0, plus_pass_rate=0.25)
        == "base_ok_plus_fail"
    )


def test_classify_score_separates_partial_from_total_failure() -> None:
    probe = _load_probe_hardness()

    assert probe.classify_score(base_pass_rate=0.0, plus_pass_rate=0.4) == "partial"
    assert probe.classify_score(base_pass_rate=0.0, plus_pass_rate=0.0) == "failed"
