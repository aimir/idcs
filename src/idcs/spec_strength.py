"""Composite spec-strength score (0–100) from a completed episode trace.

Combines several dimensions of spec quality into a single interpretable
number.  Inspired by the Kimi research's "Ringer Score" concept — a quick
summary that answers *"how battle-tested is this spec?"*.

Dimensions (weights sum to 100):

 1. **Attack survival**  (30 pts) — fraction of D→G turns that ended with
    no remaining issues.
 2. **Issue resolution**  (25 pts) — fraction of type-1 issues raised that
    were subsequently fixed (disappeared in the next turn).
 3. **Clarification value** (20 pts) — fraction of type-2 questions the
    user actually answered (not dismissed).
 4. **Spec richness**     (15 pts) — whether the spec fills the expected
    structural fields (postconditions, edge_cases, acceptance_criteria, …).
 5. **Convergence speed**  (10 pts) — bonus for converging in fewer turns
    (reaching an issue-free turn early).
"""

from __future__ import annotations

from idcs.schemas import Spec, Trace


def spec_strength(trace: Trace, *, max_turns: int = 5) -> float:
    """Return a composite spec-strength score in [0, 100].

    ``max_turns`` is the episode budget — used to score convergence speed.
    """
    return (
        _attack_survival(trace) * 30
        + _issue_resolution(trace) * 25
        + _clarification_value(trace) * 20
        + _spec_richness(trace.final_spec) * 15
        + _convergence_speed(trace, max_turns) * 10
    )


# -- dimension helpers (each returns a float in [0.0, 1.0]) ---------------


def _attack_survival(trace: Trace) -> float:
    """Fraction of turns that ended with zero issues."""
    if not trace.turns:
        return 0.0
    clean = sum(1 for t in trace.turns if not t.issues)
    return clean / len(trace.turns)


def _issue_resolution(trace: Trace) -> float:
    """Fraction of type-1 issues that disappeared by the next turn."""
    if len(trace.turns) < 2:
        return 1.0 if trace.turns and not trace.turns[0].issues else 0.0
    raised = 0
    fixed = 0
    for cur, nxt in zip(trace.turns, trace.turns[1:], strict=False):
        cur_keys = {
            _issue_sig(i) for i in cur.issues if i.route == "generator"
        }
        if not cur_keys:
            continue
        raised += len(cur_keys)
        nxt_keys = {_issue_sig(i) for i in nxt.issues}
        fixed += sum(1 for k in cur_keys if k not in nxt_keys)
    if raised == 0:
        return 1.0
    return fixed / raised


def _clarification_value(trace: Trace) -> float:
    """Fraction of type-2 questions the user answered (not dismissed)."""
    if not trace.turns:
        return 0.0
    asked = 0
    answered = 0
    for turn in trace.turns:
        for issue in turn.issues:
            if issue.route == "user":
                asked += 1
                if issue.location in turn.user_answers:
                    answered += 1
    if asked == 0:
        return 1.0  # no questions needed → full marks
    return answered / asked


def _spec_richness(spec: Spec | None) -> float:
    """Score based on how many structural fields the spec fills.

    A spec that only sets ``goal`` gets near-zero; one that fills
    postconditions, edge_cases, and acceptance_criteria scores higher.
    """
    if spec is None:
        return 0.0
    # Each field contributes equally; we count non-empty list fields.
    fields = [
        spec.inputs,
        spec.outputs,
        spec.preconditions,
        spec.postconditions,
        spec.invariants,
        spec.edge_cases,
        spec.acceptance_criteria,
    ]
    filled = sum(1 for f in fields if f)
    # goal is always present by construction, so don't count it.
    return filled / len(fields)


def _convergence_speed(trace: Trace, max_turns: int) -> float:
    """Bonus for reaching an issue-free turn early in the episode."""
    if max_turns <= 0:
        return 0.0
    for idx, turn in enumerate(trace.turns):
        if not turn.issues:
            # Converged at turn idx (0-indexed).  Earlier = higher score.
            remaining = max_turns - (idx + 1)
            return remaining / max_turns
    return 0.0  # never converged


def _issue_sig(issue: object) -> tuple[str, str, str]:
    """Lightweight identity for dedup across turns."""
    return (
        getattr(issue, "kind", ""),
        getattr(issue, "location", ""),
        getattr(issue, "description", ""),
    )
