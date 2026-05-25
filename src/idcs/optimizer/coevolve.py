"""Co-optimization loop for generator + distinguisher prompts."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from idcs.benchmark.scoring import score
from idcs.coder import Coder
from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.llm import LLMClient
from idcs.optimizer.mutate import Mutator
from idcs.optimizer.population import Population, PromptCandidate
from idcs.orchestrator import run_episode
from idcs.rewards import RewardWeights, compute_reward_breakdown
from idcs.schemas import RewardBreakdown, Task, Trace
from idcs.telemetry import create_run_dir, write_metrics, write_trace
from idcs.user_proxy import UserProxy


@dataclass
class CoevolveConfig:
    population_size: int = 8
    elite_size: int = 3
    epochs: int = 5
    max_turns: int = 3
    task_sample_size: int | None = None
    seed: int = 42
    telemetry: bool = True


@dataclass
class CoevolveResult:
    generator: Population
    distinguisher: Population
    run_dir: Path | None


def coevolve(
    tasks: list[Task],
    llm: LLMClient,
    *,
    generator_prompt: str,
    distinguisher_prompt: str,
    user_factory: Callable[[Task], UserProxy],
    weights: RewardWeights | None = None,
    config: CoevolveConfig | None = None,
) -> CoevolveResult:
    if not tasks:
        raise ValueError("tasks must be non-empty")
    weights = weights or RewardWeights()
    config = config or CoevolveConfig()
    rng = random.Random(config.seed)
    run_dir = create_run_dir() if config.telemetry else None

    mutator = Mutator(llm)
    generator_pop = _init_population(
        generator_prompt, config.population_size, mutator, "generator", rng
    )
    distinguisher_pop = _init_population(
        distinguisher_prompt, config.population_size, mutator, "distinguisher", rng
    )

    for epoch in range(1, config.epochs + 1):
        generator_pop = _evolve_population(
            role="generator",
            population=generator_pop,
            opponent_population=distinguisher_pop,
            tasks=_sample_tasks(tasks, config.task_sample_size, rng),
            llm=llm,
            user_factory=user_factory,
            weights=weights,
            config=config,
            mutator=mutator,
            rng=rng,
            run_dir=run_dir,
            epoch=epoch,
        )
        distinguisher_pop = _evolve_population(
            role="distinguisher",
            population=distinguisher_pop,
            opponent_population=generator_pop,
            tasks=_sample_tasks(tasks, config.task_sample_size, rng),
            llm=llm,
            user_factory=user_factory,
            weights=weights,
            config=config,
            mutator=mutator,
            rng=rng,
            run_dir=run_dir,
            epoch=epoch,
        )

    return CoevolveResult(generator=generator_pop, distinguisher=distinguisher_pop, run_dir=run_dir)


def _init_population(
    base_prompt: str,
    size: int,
    mutator: Mutator,
    role: str,
    rng: random.Random,
) -> Population:
    members = [PromptCandidate(prompt=base_prompt)]
    if size > 1:
        feedback = f"Seed mutations for {role} prompt diversity."
        mutations = mutator.mutate(base_prompt, feedback, count=size - 1)
        rng.shuffle(mutations)
        for prompt in mutations:
            members.append(PromptCandidate(prompt=prompt))
    while len(members) < size:
        members.append(PromptCandidate(prompt=base_prompt))
    return Population(members=members)


def _evolve_population(
    *,
    role: str,
    population: Population,
    opponent_population: Population,
    tasks: list[Task],
    llm: LLMClient,
    user_factory: Callable[[Task], UserProxy],
    weights: RewardWeights,
    config: CoevolveConfig,
    mutator: Mutator,
    rng: random.Random,
    run_dir: Path | None,
    epoch: int,
) -> Population:
    evaluated = [
        _evaluate_candidate(
            role=role,
            candidate=candidate,
            opponent_population=opponent_population,
            tasks=tasks,
            llm=llm,
            user_factory=user_factory,
            weights=weights,
            config=config,
            rng=rng,
            run_dir=run_dir,
            epoch=epoch,
        )
        for candidate in population.members
    ]
    evaluated_population = Population(members=evaluated)
    elites = evaluated_population.top_k(config.elite_size)
    if run_dir is not None and evaluated_population.members:
        best = elites[0] if elites else None
        if best is not None:
            write_metrics(
                run_dir,
                {
                    "epoch": epoch,
                    "role": role,
                    "best_reward": best.reward,
                    "avg_reward": mean(evaluated_population.rewards()),
                    "population_size": len(evaluated_population.members),
                },
            )

    new_members = list(elites)
    while len(new_members) < config.population_size:
        parent = rng.choice(elites or evaluated_population.members)
        feedback = _summarize_feedback(parent, role)
        mutations = mutator.mutate(parent.prompt, feedback, count=1)
        if mutations:
            new_members.append(PromptCandidate(prompt=mutations[0]))
        else:
            new_members.append(PromptCandidate(prompt=parent.prompt))
    return Population(members=new_members)


def _evaluate_candidate(
    *,
    role: str,
    candidate: PromptCandidate,
    opponent_population: Population,
    tasks: list[Task],
    llm: LLMClient,
    user_factory: Callable[[Task], UserProxy],
    weights: RewardWeights,
    config: CoevolveConfig,
    rng: random.Random,
    run_dir: Path | None,
    epoch: int,
) -> PromptCandidate:
    coder = Coder(llm)
    breakdowns: list[RewardBreakdown] = []
    rewards: list[float] = []
    for task in tasks:
        opponent_prompt = _sample_opponent(opponent_population, rng)
        generator_prompt = candidate.prompt if role == "generator" else opponent_prompt
        distinguisher_prompt = candidate.prompt if role == "distinguisher" else opponent_prompt

        generator = Generator(llm, prompt=generator_prompt)
        distinguisher = Distinguisher(llm, prompt=distinguisher_prompt)
        user = user_factory(task)

        trace = run_episode(task, generator, distinguisher, user, max_turns=config.max_turns)
        benchmark = _score_trace(task, trace, coder)
        trace.benchmark_score = benchmark
        breakdown = compute_reward_breakdown(
            trace,
            task.prompt,
            benchmark_score=benchmark,
            weights=weights,
        )
        trace.rewards = breakdown

        breakdowns.append(breakdown)
        rewards.append(breakdown.r_generator if role == "generator" else breakdown.r_distinguisher)
        if run_dir is not None:
            write_trace(run_dir, trace)
            write_metrics(
                run_dir,
                _trace_metrics(
                    epoch=epoch,
                    role=role,
                    task_id=task.id,
                    prompt=candidate.prompt,
                    reward=rewards[-1],
                    benchmark_score=benchmark,
                ),
            )

    candidate.reward = mean(rewards) if rewards else 0.0
    candidate.breakdowns = breakdowns
    return candidate


def _score_trace(task: Task, trace: Trace, coder: Coder) -> float:
    if trace.final_spec is None:
        return 0.0
    code = coder.from_spec(trace.final_spec, task.prompt)
    return score(task, code)


def _sample_opponent(population: Population, rng: random.Random) -> str:
    if not population.members:
        raise ValueError("opponent population is empty")
    return rng.choice(population.members).prompt


def _sample_tasks(tasks: list[Task], sample_size: int | None, rng: random.Random) -> list[Task]:
    if sample_size is None or sample_size >= len(tasks):
        return list(tasks)
    if sample_size <= 0:
        raise ValueError("task_sample_size must be positive")
    return rng.sample(tasks, sample_size)


def _summarize_feedback(candidate: PromptCandidate, role: str) -> str:
    if not candidate.breakdowns:
        return f"Improve {role} prompt to raise benchmark score and reduce issues."
    avg_benchmark = mean(b.benchmark_score for b in candidate.breakdowns)
    avg_type1 = mean(b.type1_count for b in candidate.breakdowns)
    avg_type2 = mean(b.type2_count for b in candidate.breakdowns)
    return (
        f"Role={role}. Avg benchmark={avg_benchmark:.3f}. "
        f"Avg type1={avg_type1:.2f}, avg type2={avg_type2:.2f}. "
        "Improve benchmark score and reduce avoidable issues."
    )


def _trace_metrics(
    *,
    epoch: int,
    role: str,
    task_id: str,
    prompt: str,
    reward: float,
    benchmark_score: float,
) -> dict[str, object]:
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return {
        "epoch": epoch,
        "role": role,
        "task_id": task_id,
        "prompt_hash": prompt_hash,
        "reward": reward,
        "benchmark_score": benchmark_score,
    }
