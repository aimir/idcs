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
    baseline_scores: dict[str, float] | None = None,
) -> CoevolveResult:
    if not tasks:
        raise ValueError("tasks must be non-empty")
    weights = weights or RewardWeights()
    config = config or CoevolveConfig()
    rng = random.Random(config.seed)
    run_dir = create_run_dir() if config.telemetry else None

    # Per-task no-spec baseline (direct prompt → code → score). Used as the
    # floor in useful_clarification_rate attribution. Computed once for the
    # whole training run; without this, baseline_score=None forces the
    # clarification-rate term to 0 on every candidate evaluation.
    baselines = baseline_scores if baseline_scores is not None else _compute_baselines(tasks, llm)

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
            baselines=baselines,
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
            baselines=baselines,
        )

    return CoevolveResult(generator=generator_pop, distinguisher=distinguisher_pop, run_dir=run_dir)


def _compute_baselines(tasks: list[Task], llm: LLMClient) -> dict[str, float]:
    """Score ``coder.from_prompt`` once per task.

    Cost: one LLM call + one ``score()`` per task. Linear in |tasks|, not
    multiplied by population × epochs × roles.
    """
    coder = Coder(llm)
    return {task.id: score(task, coder.from_prompt(task.prompt)) for task in tasks}


def _init_population(
    base_prompt: str,
    size: int,
    mutator: Mutator,
    role: str,
    rng: random.Random,
) -> Population:
    members = [PromptCandidate(prompt=base_prompt)]
    if size > 1:
        feedback = (
            "Seed mutations for diversity. No evaluation data yet — produce "
            "variants that differ in structure or emphasis, not paraphrases."
        )
        mutations = mutator.mutate(base_prompt, feedback, role=role, count=size - 1)
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
    baselines: dict[str, float],
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
            baselines=baselines,
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
        mutations = mutator.mutate(parent.prompt, feedback, role=role, count=1)
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
    baselines: dict[str, float],
) -> PromptCandidate:
    coder = Coder(llm)
    # One opponent per candidate evaluation, not per task. Per-task sampling
    # adds noise that's not meaningful — we want this candidate's average
    # reward against *one* sampled opponent, not against a random walk through
    # the opponent population.
    opponent_prompt = _sample_opponent(opponent_population, rng)
    breakdowns: list[RewardBreakdown] = []
    rewards: list[float] = []
    for task in tasks:
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
            baseline_score=baselines.get(task.id),
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
    """Role-specific feedback string fed into Mutator.mutate(...)."""
    if not candidate.breakdowns:
        return (
            f"No evaluation data yet for this {role} prompt. "
            "Produce a variant that differs in structure or emphasis."
        )
    n = len(candidate.breakdowns)
    avg_benchmark = mean(b.benchmark_score for b in candidate.breakdowns)
    avg_type1 = mean(b.type1_count for b in candidate.breakdowns)
    avg_type2 = mean(b.type2_count for b in candidate.breakdowns)
    avg_type1_fixed = mean(b.type1_fixed_count for b in candidate.breakdowns)
    avg_type2_dismissed = mean(b.type2_dismissed_count for b in candidate.breakdowns)
    avg_useful_rate = mean(b.useful_clarification_rate for b in candidate.breakdowns)
    avg_spec_penalty = mean(b.spec_complexity_penalty for b in candidate.breakdowns)

    if role == "generator":
        return (
            f"Generator results over {n} tasks. "
            f"avg benchmark={avg_benchmark:.3f} (higher is better). "
            f"avg type-1 issues D raised against your specs={avg_type1:.2f} "
            f"(lower is better — these are gaps you should have caught). "
            f"avg spec complexity penalty={avg_spec_penalty:.3f} "
            f"(avoid empty / thin specs). Improve by producing specs "
            f"concrete enough that D has fewer gaps to flag, without "
            f"dropping benchmark score."
        )
    return (
        f"Distinguisher results over {n} tasks. "
        f"avg benchmark={avg_benchmark:.3f}. "
        f"avg type-1 issues you raised={avg_type1:.2f}; "
        f"of those, avg actually fixed next turn={avg_type1_fixed:.2f} "
        f"(this is your accepted-reject rate — higher is better). "
        f"avg type-2 questions={avg_type2:.2f}; "
        f"avg dismissed by user={avg_type2_dismissed:.2f} "
        f"(lower is better). useful clarification rate={avg_useful_rate:.3f} "
        f"(positive means your questions improved the spec). "
        f"Improve by raising issues G will accept and asking only "
        f"genuinely useful user-routed questions (cap is 5/episode)."
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
