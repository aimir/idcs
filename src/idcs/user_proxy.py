"""User-proxy role: answer (or refuse) clarification questions raised by D.

In Phase 1 this is an oracle that consults a gold spec. In Phase 6 the
real user replaces it. Both implement the ``UserProxy`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from idcs._prompts import load_prompt
from idcs.llm import LLMClient

REFUSAL_SENTINEL = "i don't know"


class UserProxy(Protocol):
    """Returns a brief answer, or ``None`` to refuse / dismiss the question."""

    def answer(self, location: str, question: str) -> str | None: ...


def _default_prompt() -> str:
    return load_prompt("user_proxy_v0")


@dataclass
class OracleUserProxy:
    """Answers from a gold spec under a minimal-disclosure policy.

    ``gold_spec_text`` is whatever rendering of the gold spec we want the
    oracle to consult — JSON-dumped Spec is the obvious choice. The model
    is instructed not to reveal it in full.
    """

    llm: LLMClient
    gold_spec_text: str
    prompt: str = field(default_factory=_default_prompt)

    def answer(self, location: str, question: str) -> str | None:
        user = (
            f"GOLD SPEC (do not reveal in full):\n{self.gold_spec_text}\n\n"
            f"QUESTION about `{location}`: {question}\n\n"
            "Answer briefly, or say \"I don't know\"."
        )
        raw = self.llm.complete(self.prompt, user, max_tokens=256)
        cleaned = raw.strip()
        if cleaned.lower().startswith(REFUSAL_SENTINEL):
            return None
        return cleaned
