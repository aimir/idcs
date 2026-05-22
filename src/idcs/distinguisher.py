"""Distinguisher role: (task, spec) → list[Issue]."""

from __future__ import annotations

from dataclasses import dataclass, field

from idcs._prompts import load_prompt
from idcs.llm import LLMClient
from idcs.schemas import Issue, IssueList, Spec


def _default_prompt() -> str:
    return load_prompt("distinguisher_v0")


@dataclass
class Distinguisher:
    """One LLM with one fixed system prompt."""

    llm: LLMClient
    prompt: str = field(default_factory=_default_prompt)

    def critique(self, task_prompt: str, spec: Spec) -> list[Issue]:
        user = (
            f"TASK:\n{task_prompt}\n\n"
            f"PROPOSED SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
            "List substantive issues, or return an empty list."
        )
        result = self.llm.complete_typed(self.prompt, user, IssueList)
        return result.issues
