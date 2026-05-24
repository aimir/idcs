"""Coder role: Spec → Python code, or task.prompt → Python code (direct mode)."""

from __future__ import annotations

from dataclasses import dataclass, field

from idcs._prompts import load_prompt
from idcs.llm import LLMClient
from idcs.schemas import Spec


def _default_prompt() -> str:
    return load_prompt("coder_v0")


@dataclass
class Coder:
    """Generates Python source from a Spec or a raw task prompt."""

    llm: LLMClient
    prompt: str = field(default_factory=_default_prompt)

    def from_spec(self, spec: Spec, task_prompt: str) -> str:
        user = (
            "MODE: from_spec\n\n"
            f"TASK DESCRIPTION:\n{task_prompt}\n\n"
            f"STRUCTURED SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
            "Produce the Python function."
        )
        return _extract_code(self.llm.complete(self.prompt, user))

    def from_prompt(self, task_prompt: str) -> str:
        user = (
            "MODE: from_prompt\n\n"
            f"TASK DESCRIPTION:\n{task_prompt}\n\n"
            "Produce the Python function."
        )
        return _extract_code(self.llm.complete(self.prompt, user))


def _extract_code(raw: str) -> str:
    """Strip markdown fences if the LLM wraps output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
