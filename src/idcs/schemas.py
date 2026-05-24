"""Core data types shared across the pipeline.

Mirrors the abstractions in docs/design.md. Pydantic v2 for JSON round-trip,
structural validation, and IDE support. Schemas are deliberately permissive —
the structure is informational, not enforced (a `type` field is free-form text,
not a runtime check).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field as PydField


class Test(BaseModel):
    """A single benchmark test case. Hidden from G and D."""

    __test__ = False  # pytest: not a test class

    id: str
    code: str


class Task(BaseModel):
    """One benchmark item: NL prompt + hidden test suite."""

    id: str
    prompt: str
    entry_point: str | None = None
    tests: list[Test]


class Field(BaseModel):
    """One named input or output in a spec."""

    name: str
    type: str
    description: str
    constraints: list[str] = PydField(default_factory=list)


class Spec(BaseModel):
    """Structured spec the generator produces."""

    goal: str
    inputs: list[Field] = PydField(default_factory=list)
    outputs: list[Field] = PydField(default_factory=list)
    preconditions: list[str] = PydField(default_factory=list)
    postconditions: list[str] = PydField(default_factory=list)
    invariants: list[str] = PydField(default_factory=list)
    edge_cases: list[str] = PydField(default_factory=list)
    acceptance_criteria: list[str] = PydField(default_factory=list)


IssueKind = Literal[
    "gap",
    "ambiguity",
    "contradiction",
    "over_constraint",
    "underconstraint",
    "implicit_assumption",
]
IssueRoute = Literal["generator", "user"]


class Issue(BaseModel):
    """One gap or ambiguity raised by the distinguisher.

    `route="generator"` is a type-1 reject (G should fix without bothering the user).
    `route="user"` is a type-2 reject (worth asking the user about).
    """

    kind: IssueKind
    route: IssueRoute
    location: str
    description: str
    suggested_question: str | None = None


class Turn(BaseModel):
    """One iteration of the G → D → route loop."""

    spec: Spec
    issues: list[Issue] = PydField(default_factory=list)
    user_answers: dict[str, str] = PydField(default_factory=dict)


class RewardBreakdown(BaseModel):
    """Per-term decomposition of R_G and R_D for one episode."""

    benchmark_score: float = 0.0
    type1_count: int = 0
    type1_fixed_count: int = 0
    type2_count: int = 0
    type2_dismissed_count: int = 0
    useful_clarification_rate: float = 0.0
    spec_complexity_penalty: float = 0.0
    benchmark_delta: float = 0.0
    regression_penalty: float = 0.0
    r_generator: float = 0.0
    r_distinguisher: float = 0.0


class Trace(BaseModel):
    """Full episode record. Written to experiments/runs/<timestamp>/ as JSONL."""

    task_id: str
    turns: list[Turn] = PydField(default_factory=list)
    final_spec: Spec | None = None
    benchmark_score: float = 0.0
    rewards: RewardBreakdown = PydField(default_factory=RewardBreakdown)
    # Prompt fingerprints for cross-referencing traces ↔ candidates. Set by
    # the optimizer when a candidate is evaluated; harmless on cold-start
    # traces that don't come from a coevolution run.
    generator_prompt_hash: str | None = None
    distinguisher_prompt_hash: str | None = None


class IssueList(BaseModel):
    """Distinguisher output. Wrapped because structured output is one object, not an array."""

    issues: list[Issue] = PydField(default_factory=list)
