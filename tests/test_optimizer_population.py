from __future__ import annotations

from idcs.optimizer.population import Population, PromptCandidate


def test_population_top_k() -> None:
    pop = Population(
        members=[
            PromptCandidate(prompt="alpha system prompt", reward=0.1),
            PromptCandidate(prompt="beta system prompt", reward=0.9),
            PromptCandidate(prompt="gamma system prompt", reward=0.5),
        ]
    )
    top = pop.top_k(2)
    assert [candidate.prompt for candidate in top] == [
        "beta system prompt",
        "gamma system prompt",
    ]


def test_population_best() -> None:
    pop = Population(members=[PromptCandidate(prompt="a", reward=0.2)])
    assert pop.best().prompt == "a"


def test_top_k_skips_near_duplicates() -> None:
    """Near-identical prompts shouldn't both make it into the elite set."""
    base = (
        "You are a generator. Produce a structured spec from the task "
        "description. Be concrete and cover edge cases."
    )
    # Trailing whitespace only — SequenceMatcher.ratio() will be > 0.92.
    near_clone = base + " "
    different = (
        "You are a critic. Examine the spec for missing constraints, "
        "ambiguous terms, and over-constraints. Return a list of issues."
    )
    pop = Population(
        members=[
            PromptCandidate(prompt=base, reward=0.9),
            PromptCandidate(prompt=near_clone, reward=0.85),
            PromptCandidate(prompt=different, reward=0.7),
        ]
    )
    top = pop.top_k(2)
    prompts = [c.prompt for c in top]
    assert base in prompts
    assert different in prompts
    assert near_clone not in prompts


def test_top_k_tops_up_when_diversity_starves_pool() -> None:
    """If diversity rejects too many, caller still gets k candidates."""
    pop = Population(
        members=[
            PromptCandidate(prompt="Shared prefix with tiny ending one.", reward=0.9),
            PromptCandidate(prompt="Shared prefix with tiny ending two.", reward=0.8),
            PromptCandidate(prompt="Shared prefix with tiny ending three.", reward=0.7),
        ]
    )
    # All three are similar; diversity would leave us with only the top one.
    # We still want 3 returned — fall back to highest-reward order.
    top = pop.top_k(3)
    assert len(top) == 3


def test_top_k_diversity_threshold_1_only_rejects_exact_duplicates() -> None:
    pop = Population(
        members=[
            PromptCandidate(prompt="same", reward=0.9),
            PromptCandidate(prompt="same", reward=0.8),
            PromptCandidate(prompt="diff", reward=0.5),
        ]
    )
    top = pop.top_k(2, diversity_threshold=1.0)
    prompts = [c.prompt for c in top]
    # Exact duplicate filtered → one "same" survives, then "diff".
    assert prompts.count("same") == 1
    assert "diff" in prompts
