from __future__ import annotations

from pydantic import BaseModel

from idcs.optimizer.mutate import Mutator
from tests.fakes import FakeLLM


def test_mutator_falls_back_to_plain_text_when_structured_output_fails() -> None:
    def typed_responder(
        system: str,
        user: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        assert system
        assert "ROLE: generator" in user
        assert output_type.__name__ == "MutationBatch"
        raise RuntimeError("malformed mutation JSON")

    llm = FakeLLM(
        typed_responder=typed_responder,
        text_responses=[
            "# Generator v1\n\nPrefer explicit edge cases.",
            "# Generator v2\n\nPrefer concrete postconditions.",
        ],
    )

    prompts = Mutator(llm).mutate(
        "base prompt",
        "feedback",
        role="generator",
        count=2,
    )

    assert prompts == [
        "# Generator v1\n\nPrefer explicit edge cases.",
        "# Generator v2\n\nPrefer concrete postconditions.",
    ]
    assert len(llm.typed_calls) == 1
    assert len(llm.text_calls) == 2
    assert "alternative prompt 1 of 2" in llm.text_calls[0][1]
    assert "alternative prompt 2 of 2" in llm.text_calls[1][1]
