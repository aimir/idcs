"""Population helpers for prompt search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from idcs.schemas import RewardBreakdown

DEFAULT_DIVERSITY_THRESHOLD = 0.92
"""SequenceMatcher.ratio() above which two prompts count as near-duplicates.

0.92 rejects clones and trivial paraphrases while letting in real variants.
``Population.top_k`` exposes this as a kwarg; set to 1.0 to disable.
"""


@dataclass
class PromptCandidate:
    prompt: str
    reward: float = 0.0
    breakdowns: list[RewardBreakdown] = field(default_factory=list)


@dataclass
class Population:
    members: list[PromptCandidate]

    def top_k(
        self,
        k: int,
        *,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    ) -> list[PromptCandidate]:
        """Top-k by reward, enforcing pairwise prompt diversity.

        Walks candidates in descending-reward order and skips any whose
        prompt is too similar (per ``SequenceMatcher.ratio()``) to one
        already selected. If diversity filtering leaves fewer than ``k``
        candidates, top up from the rejects so the caller still gets ``k``.
        """
        if k <= 0:
            return []
        sorted_members = sorted(self.members, key=lambda c: c.reward, reverse=True)
        selected: list[PromptCandidate] = []
        rejected: list[PromptCandidate] = []
        for member in sorted_members:
            if len(selected) >= k:
                rejected.append(member)
                continue
            if any(
                _too_similar(member.prompt, picked.prompt, diversity_threshold)
                for picked in selected
            ):
                rejected.append(member)
                continue
            selected.append(member)
        if len(selected) < k:
            selected.extend(rejected[: k - len(selected)])
        return selected

    def best(self) -> PromptCandidate:
        if not self.members:
            raise ValueError("population is empty")
        return max(self.members, key=lambda c: c.reward)

    def rewards(self) -> list[float]:
        return [member.reward for member in self.members]

    def extend(self, candidates: Iterable[PromptCandidate]) -> None:
        self.members.extend(list(candidates))


def _too_similar(a: str, b: str, threshold: float) -> bool:
    if threshold >= 1.0:
        return a == b
    return SequenceMatcher(None, a, b).ratio() >= threshold
