"""Regression tests for the curated hardened-problem corpus."""

from __future__ import annotations

import json

from idcs.benchmark.scoring import score
from idcs.benchmark.tasks import load_benchmark_tasks, load_hardened_items

KNOWN_GOOD = {
    "hardened/001-normalize-ranges": """
def normalize_ranges(ranges):
    normalized = []
    for start, end in ranges:
        lo, hi = (start, end) if start <= end else (end, start)
        normalized.append((lo, hi))
    normalized.sort()
    merged = []
    for lo, hi in normalized:
        if not merged or lo > merged[-1][1] + 1:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]
""",
    "hardened/002-unique-once": """
from collections import Counter

def unique_values(values):
    counts = Counter(values)
    return [value for value in values if counts[value] == 1]
""",
    "hardened/003-apply-coupon": """
def apply_coupon(subtotal_cents, coupon):
    if not isinstance(coupon, dict) or "type" not in coupon or "value" not in coupon:
        return subtotal_cents
    if subtotal_cents < coupon.get("min_subtotal_cents", 0):
        return subtotal_cents
    if coupon["type"] == "percent":
        discount = subtotal_cents * coupon["value"] // 100
    elif coupon["type"] == "fixed":
        discount = coupon["value"]
    else:
        return subtotal_cents
    if "max_discount_cents" in coupon:
        discount = min(discount, coupon["max_discount_cents"])
    return max(0, subtotal_cents - discount)
""",
    "hardened/004-redact-tokens": r"""
import re

def redact_tokens(text):
    return re.sub(r"(?<!\S)sk-[A-Za-z0-9-]{8,}", "[REDACTED]", text)
""",
    "hardened/005-primary-contact": """
def pick_primary_contact(contacts):
    best = None
    best_key = None
    for index, contact in enumerate(contacts):
        email = contact.get("email")
        if contact.get("verified") is not True or not isinstance(email, str) or not email:
            continue
        key = (contact.get("priority", 0), -contact.get("updated_at", 0), index)
        if best_key is None or key < best_key:
            best = email
            best_key = key
    return best
""",
}


def test_hardened_corpus_has_tasks_specs_and_weakness_notes() -> None:
    items = load_hardened_items()

    assert len(items) >= 5
    assert all(item.task.id.startswith("hardened/") for item in items)
    assert all(item.task.entry_point for item in items)
    assert all(item.task.tests for item in items)
    assert all(item.gold_spec.goal for item in items)
    assert all(item.known_weakness.get("type") for item in items)


def test_hardened_dataset_loads_as_benchmark_tasks() -> None:
    tasks = load_benchmark_tasks("hardened")

    assert [task.id for task in tasks] == [item.task.id for item in load_hardened_items()]


def test_hardened_corpus_accepts_known_good_solutions() -> None:
    for item in load_hardened_items():
        assert score(item.task, KNOWN_GOOD[item.task.id]) == 1.0, item.task.id


def test_hardened_loader_rejects_missing_gold_spec(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"task": {"id": "hardened/bad", "prompt": "", "tests": []}}))

    try:
        load_hardened_items(tmp_path)
    except ValueError as exc:
        assert "gold_spec" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hardened_loader_accepts_nested_candidate_directories(tmp_path) -> None:
    nested = tmp_path / "policy"
    nested.mkdir()
    (nested / "candidate.json").write_text(
        json.dumps(
            {
                "task": {
                    "id": "candidate/policy/001",
                    "prompt": "Write f.",
                    "entry_point": "f",
                    "tests": [{"id": "candidate/policy/001/t0", "code": "assert f() == 1"}],
                },
                "gold_spec": {
                    "goal": "Return one.",
                    "acceptance_criteria": ["f() == 1"],
                },
                "known_weakness": {"type": "ambiguity"},
            }
        ),
        encoding="utf-8",
    )

    tasks = load_benchmark_tasks("hardened", hardened_dir=tmp_path)
    items = load_hardened_items(tmp_path)

    assert [task.id for task in tasks] == ["candidate/policy/001"]
    assert items[0].path == nested / "candidate.json"
