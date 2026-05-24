"""LLM-driven prompt mutation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from idcs.llm import LLMClient

_DEFAULT_MUTATOR_SYSTEM = """You improve system prompts in an adversarial
spec-generation pipeline. Two roles share this loop:

- **generator** — translates a natural-language task description into a
  structured specification (goal, inputs, outputs, pre/postconditions,
  edge cases, acceptance criteria).
- **distinguisher** — critiques a spec against the task, returning issues
  classified as gap / ambiguity / contradiction / over_constraint /
  underconstraint / implicit_assumption, each routed to either the
  generator (auto-fixable) or the user (needs clarification).

You receive ONE role's current prompt plus feedback summarizing how it
performed in evaluation. Produce N alternative prompts.

Rules:
- Preserve the role's core function. A generator prompt must still
  produce specs; a distinguisher prompt must still critique them.
- Address the feedback. The feedback names specific failure modes —
  your edits should plausibly improve them.
- Be substantive. Don't paraphrase — make a real change (add a
  principle, drop a constraint, restructure a section).
- Keep markdown structure (`#`, `##` headings, bullet lists) where the
  base prompt has it.

Return only a JSON object matching the schema in the user message.
"""


class MutationBatch(BaseModel):
    prompts: list[str] = Field(default_factory=list)


@dataclass
class Mutator:
    llm: LLMClient
    system_prompt: str = _DEFAULT_MUTATOR_SYSTEM

    def mutate(
        self,
        base_prompt: str,
        feedback: str,
        *,
        role: str = "generator",
        count: int = 3,
        max_tokens: int = 2000,
    ) -> list[str]:
        """Return up to ``count`` mutated variants of ``base_prompt``."""
        user = (
            f"ROLE: {role}\n\n"
            f"BASE PROMPT:\n{base_prompt}\n\n"
            f"FEEDBACK FROM EVALUATION:\n{feedback}\n\n"
            f"Produce {count} alternative prompts that address the feedback."
        )
        result = self.llm.complete_typed(
            self.system_prompt, user, MutationBatch, max_tokens=max_tokens
        )
        cleaned = [prompt.strip() for prompt in result.prompts if prompt.strip()]
        return cleaned[:count]
