from __future__ import annotations

from idcs.optimizer.population import Population, PromptCandidate


def test_population_top_k() -> None:
    pop = Population(
        members=[
            PromptCandidate(prompt="a", reward=0.1),
            PromptCandidate(prompt="b", reward=0.9),
            PromptCandidate(prompt="c", reward=0.5),
        ]
    )
    top = pop.top_k(2)
    assert [candidate.prompt for candidate in top] == ["b", "c"]


def test_population_best() -> None:
    pop = Population(members=[PromptCandidate(prompt="a", reward=0.2)])
    assert pop.best().prompt == "a"
