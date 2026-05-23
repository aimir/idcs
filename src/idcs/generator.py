"""Generator role: NL task → Spec, and (spec + issues + answers) → revised Spec."""

from __future__ import annotations

from dataclasses import dataclass, field

from idcs._prompts import load_prompt
from idcs.llm import LLMClient
from idcs.schemas import Issue, Spec


def _default_prompt() -> str:
    return load_prompt("generator_v0")


@dataclass
class Generator:
    """One LLM with one fixed system prompt."""

    llm: LLMClient
    prompt: str = field(default_factory=_default_prompt)

    def draft(self, task_prompt: str) -> Spec:
        """Initial spec from a task description."""
        user = (
            "MODE: draft\n\n"
            f"TASK:\n{task_prompt}\n\n"
            "Produce the structured spec."
        )
        return self.llm.complete_typed(self.prompt, user, Spec)

    def revise(
        self,
        task_prompt: str,
        spec: Spec,
        issues: list[Issue],
        answers: dict[str, str],
    ) -> Spec:
        """Revised spec given the current spec, identified issues, and user answers."""
        rendered_issues = "\n".join(
            f"- [{i.kind}, route={i.route}, at {i.location}] {i.description}"
            + (
                f" | answer: {answers[i.location]}"
                if i.route == "user" and i.location in answers
                else ""
            )
            for i in issues
        )
        user = (
            "MODE: revise\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"CURRENT SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
            f"ISSUES:\n{rendered_issues}\n\n"
            "Produce the revised spec."
        )
        return self.llm.complete_typed(self.prompt, user, Spec)
