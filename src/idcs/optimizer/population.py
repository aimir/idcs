"""Population helpers for prompt search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from idcs.schemas import RewardBreakdown


@dataclass
class PromptCandidate:
    prompt: str
    reward: float = 0.0
    breakdowns: list[RewardBreakdown] = field(default_factory=list)


@dataclass
class Population:
    members: list[PromptCandidate]

    def top_k(self, k: int) -> list[PromptCandidate]:
        if k <= 0:
            return []
        return sorted(self.members, key=lambda c: c.reward, reverse=True)[:k]

    def best(self) -> PromptCandidate:
        if not self.members:
            raise ValueError("population is empty")
        return max(self.members, key=lambda c: c.reward)

    def rewards(self) -> list[float]:
        return [member.reward for member in self.members]

    def extend(self, candidates: Iterable[PromptCandidate]) -> None:
        self.members.extend(list(candidates))
