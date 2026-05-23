"""LLM-driven prompt mutation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from idcs.llm import LLMClient

_DEFAULT_MUTATOR_SYSTEM = """You improve prompts for a prompt-optimization system.
Generate concise edits that preserve the role but improve performance.
Return only the JSON object requested by the user."""


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
        count: int = 3,
        max_tokens: int = 2000,
    ) -> list[str]:
        user = (
            f"BASE PROMPT:\n{base_prompt}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            f"Return {count} improved prompts."
        )
        result = self.llm.complete_typed(self.system_prompt, user, MutationBatch, max_tokens=max_tokens)
        cleaned = [prompt.strip() for prompt in result.prompts if prompt.strip()]
        return cleaned[:count]
