"""Co-optimization loop for generator + distinguisher prompts."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from idcs.benchmark.scoring import FailureExample, ScoreResult, score, score_detailed
from idcs.coder import Coder
from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.llm import BudgetExceededError, LLMClient
from idcs.optimizer.mutate import Mutator
from idcs.optimizer.population import Population, PromptCandidate
from idcs.orchestrator import run_episode
from idcs.rewards import RewardWeights, compute_reward_breakdown
from idcs.schemas import RewardBreakdown, Task, Trace
from idcs.telemetry import create_run_dir, write_metrics, write_trace
from idcs.user_proxy import UserProxy

log = logging.getLogger(__name__)


def _calls_so_far(llm: LLMClient) -> str:
    """Render the running LLM call count if the client tracks it."""
    n = getattr(llm, "calls_made", None)
    return f" [{n} calls]" if n is not None else ""


@dataclass
class CoevolveConfig:
    population_size: int = 8
    elite_size: int = 3
    elite_selection: Literal["task_pareto", "mean"] = "task_pareto"
    keep_anchor: bool = True
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
    val_tasks: list[Task] | None = None,
    mutator_llm: LLMClient | None = None,
) -> CoevolveResult:
    """Run the alternating G / D evolution.

    ``tasks`` is the *training* set. ``val_tasks`` is an optional held-out
    set; when provided, the best-of-population G and D are evaluated on it
    at the end of each epoch and the results land in ``metrics.jsonl`` as
    ``val_*`` rows. Val is monitor-only — it does **not** drive selection.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty")
    weights = weights or RewardWeights()
    config = config or CoevolveConfig()
    if config.elite_size >= config.population_size:
        capped = max(1, config.population_size - 1)
        log.warning(
            "elite_size (%d) >= population_size (%d) leaves no slots for "
            "mutations between epochs — capping elite_size to %d so the "
            "loop can actually evolve.",
            config.elite_size,
            config.population_size,
            capped,
        )
        config = dataclasses.replace(config, elite_size=capped)
    rng = random.Random(config.seed)
    run_dir = create_run_dir() if config.telemetry else None

    # Per-task no-spec baseline (direct prompt → code → score). Used as the
    # floor in useful_clarification_rate attribution. Computed once for the
    # whole training run; without this, baseline_score=None forces the
    # clarification-rate term to 0 on every candidate evaluation.
    log.info(
        "coevolve start: pop=%d elite=%d epochs=%d max_turns=%d "
        "tasks_train=%d tasks_val=%d task_sample=%s",
        config.population_size,
        config.elite_size,
        config.epochs,
        config.max_turns,
        len(tasks),
        len(val_tasks or []),
        config.task_sample_size,
    )
    baseline_targets = list(tasks) + list(val_tasks or [])
    if baseline_scores is None:
        log.info(
            "computing no-spec baselines for %d tasks...",
            len(baseline_targets),
        )
    baselines = (
        baseline_scores
        if baseline_scores is not None
        else _compute_baselines(baseline_targets, llm)
    )
    if baselines:
        bvals = list(baselines.values())
        log.info(
            "baselines: min=%.3f mean=%.3f max=%.3f n=%d%s",
            min(bvals), mean(bvals), max(bvals), len(bvals), _calls_so_far(llm),
        )

    if run_dir is not None:
        _write_config_snapshot(
            run_dir,
            llm=llm,
            mutator_llm=mutator_llm,
            config=config,
            weights=weights,
            train_tasks=tasks,
            val_tasks=val_tasks,
            baselines=baselines,
        )

    mutator = Mutator(mutator_llm if mutator_llm is not None else llm)
    log.info("seeding initial populations via mutator...")
    generator_pop = _init_population(
        generator_prompt, config.population_size, mutator, "generator", rng
    )
    distinguisher_pop = _init_population(
        distinguisher_prompt, config.population_size, mutator, "distinguisher", rng
    )
    log.info("initial populations ready%s", _calls_so_far(llm))

    for epoch in range(1, config.epochs + 1):
        log.info("=== epoch %d/%d ===", epoch, config.epochs)
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
        log.info(
            "epoch %d generator: best=%.3f avg=%.3f%s",
            epoch,
            generator_pop.best().reward,
            mean(generator_pop.rewards()),
            _calls_so_far(llm),
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
        log.info(
            "epoch %d distinguisher: best=%.3f avg=%.3f%s",
            epoch,
            distinguisher_pop.best().reward,
            mean(distinguisher_pop.rewards()),
            _calls_so_far(llm),
        )

        if val_tasks and run_dir is not None:
            log.info("epoch %d val eval (%d tasks)...", epoch, len(val_tasks))
            val_metrics = _evaluate_on_val(
                g_prompt=generator_pop.best().prompt,
                d_prompt=distinguisher_pop.best().prompt,
                val_tasks=val_tasks,
                llm=llm,
                user_factory=user_factory,
                weights=weights,
                max_turns=config.max_turns,
                baselines=baselines,
            )
            log.info(
                "epoch %d val: benchmark=%.3f rG=%.3f rD=%.3f (n=%d)%s",
                epoch,
                val_metrics["val_avg_benchmark"],
                val_metrics["val_avg_r_generator"],
                val_metrics["val_avg_r_distinguisher"],
                int(val_metrics["val_n_tasks"]),
                _calls_so_far(llm),
            )
            write_metrics(
                run_dir,
                {"epoch": epoch, "split": "val", **val_metrics},
            )

    return CoevolveResult(generator=generator_pop, distinguisher=distinguisher_pop, run_dir=run_dir)


def _evaluate_on_val(
    *,
    g_prompt: str,
    d_prompt: str,
    val_tasks: list[Task],
    llm: LLMClient,
    user_factory: Callable[[Task], UserProxy],
    weights: RewardWeights,
    max_turns: int,
    baselines: dict[str, float],
) -> dict[str, float]:
    """Best-G vs best-D on the val split. Monitor-only; never feeds selection."""
    coder = Coder(llm)
    g = Generator(llm, prompt=g_prompt)
    d = Distinguisher(llm, prompt=d_prompt)
    benchmarks: list[float] = []
    g_rewards: list[float] = []
    d_rewards: list[float] = []
    for task in val_tasks:
        user = user_factory(task)
        trace = run_episode(task, g, d, user, max_turns=max_turns)
        benchmark = _score_trace(task, trace, coder)
        breakdown = compute_reward_breakdown(
            trace,
            task.prompt,
            benchmark_score=benchmark,
            weights=weights,
            baseline_score=baselines.get(task.id),
        )
        benchmarks.append(benchmark)
        g_rewards.append(breakdown.r_generator)
        d_rewards.append(breakdown.r_distinguisher)
    return {
        "val_avg_benchmark": mean(benchmarks) if benchmarks else 0.0,
        "val_avg_r_generator": mean(g_rewards) if g_rewards else 0.0,
        "val_avg_r_distinguisher": mean(d_rewards) if d_rewards else 0.0,
        "val_n_tasks": len(val_tasks),
        "llm_structured_fallback_count": float(
            getattr(llm, "structured_fallback_count", 0)
        ),
    }


def _write_config_snapshot(
    run_dir: Path,
    *,
    llm: LLMClient,
    mutator_llm: LLMClient | None,
    config: CoevolveConfig,
    weights: RewardWeights,
    train_tasks: list[Task],
    val_tasks: list[Task] | None,
    baselines: dict[str, float],
) -> None:
    """Pin the experiment's inputs in run_dir/config.json.

    Future-you reading this run wants to know: which model, which weights,
    which tasks, which baselines. Without this snapshot, traces and metrics
    are uninterpretable a week later.
    """
    main_model = getattr(llm, "model", None)
    snapshot = {
        "model": main_model,
        "mutator_model": (
            getattr(mutator_llm, "model", None) if mutator_llm is not None else main_model
        ),
        "weights": dataclasses.asdict(weights),
        "config": dataclasses.asdict(config),
        "train_task_ids": [t.id for t in train_tasks],
        "val_task_ids": [t.id for t in (val_tasks or [])],
        "baselines": baselines,
    }
    (run_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


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
    members = [PromptCandidate(prompt=base_prompt, anchor=True)]
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
    log.info(
        "  evolving %s: %d candidates × %d tasks",
        role,
        len(population.members),
        len(tasks),
    )
    evaluated: list[PromptCandidate] = []
    for i, candidate in enumerate(population.members, 1):
        log.info("    %s candidate %d/%d%s", role, i, len(population.members), _calls_so_far(llm))
        evaluated.append(
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
        )
    evaluated_population = Population(members=evaluated)
    elites = _select_elites(evaluated_population, role=role, config=config)
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
                    "elite_count": len(elites),
                    "elite_selection": config.elite_selection,
                    "frontier_task_ids": sorted(
                        {
                            task_id
                            for elite in elites
                            for task_id in elite.frontier_task_ids
                        }
                    ),
                    "llm_structured_fallback_count": getattr(
                        llm,
                        "structured_fallback_count",
                        0,
                    ),
                },
            )
        _write_population_snapshot(run_dir, epoch=epoch, role=role, population=evaluated_population)

    new_members = list(elites)
    while len(new_members) < config.population_size:
        parent = rng.choice(elites or evaluated_population.members)
        feedback = _summarize_feedback(parent, role, peer_elites=elites)
        mutations = mutator.mutate(parent.prompt, feedback, role=role, count=1)
        if mutations:
            new_members.append(PromptCandidate(prompt=mutations[0]))
        else:
            new_members.append(PromptCandidate(prompt=parent.prompt))
    return Population(members=new_members)


def _select_elites(
    population: Population,
    *,
    role: str,
    config: CoevolveConfig,
) -> list[PromptCandidate]:
    """Select survivors for the next epoch.

    ``mean`` keeps the older behavior: top-k by average reward. ``task_pareto``
    keeps prompts that are best on at least one evaluated task, so a specialist
    prompt is not discarded only because another prompt has a smoother average.
    """
    for candidate in population.members:
        candidate.frontier_task_ids = []
    if config.elite_selection == "mean":
        return _retain_anchor_elites(
            population,
            population.top_k(config.elite_size),
            config=config,
        )
    if config.elite_selection != "task_pareto":
        raise ValueError(f"unknown elite selection strategy: {config.elite_selection}")

    frontier = _task_frontier(population, role=role)
    if not frontier:
        return _retain_anchor_elites(
            population,
            population.top_k(config.elite_size),
            config=config,
        )

    # Pareto can find more useful specialists than the nominal elite size.
    # Keep them when there is room, but always leave at least one mutation slot.
    max_elites = max(1, config.population_size - 1)
    target = min(max_elites, max(config.elite_size, len(frontier)))
    selected = _rank_frontier(frontier)[:target]

    if len(selected) < config.elite_size:
        selected_ids = {id(candidate) for candidate in selected}
        for candidate in population.top_k(config.elite_size):
            if id(candidate) not in selected_ids:
                selected.append(candidate)
                selected_ids.add(id(candidate))
            if len(selected) >= config.elite_size:
                break

    return sorted(
        _retain_anchor_elites(population, selected, config=config),
        key=lambda candidate: candidate.reward,
        reverse=True,
    )


def _retain_anchor_elites(
    population: Population,
    selected: list[PromptCandidate],
    *,
    config: CoevolveConfig,
) -> list[PromptCandidate]:
    if not config.keep_anchor:
        return selected
    anchors = [candidate for candidate in population.members if candidate.anchor]
    if not anchors:
        return selected

    selected_ids = {id(candidate) for candidate in selected}
    max_elites = max(1, config.population_size - 1)
    out = list(selected)
    for anchor in anchors:
        if id(anchor) in selected_ids:
            continue
        if len(out) < max_elites:
            out.append(anchor)
            selected_ids.add(id(anchor))
            continue
        replace_index = _lowest_reward_non_anchor_index(out)
        if replace_index is not None:
            selected_ids.discard(id(out[replace_index]))
            out[replace_index] = anchor
            selected_ids.add(id(anchor))
    return out


def _lowest_reward_non_anchor_index(candidates: list[PromptCandidate]) -> int | None:
    indexed = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if not candidate.anchor
    ]
    if not indexed:
        return None
    return min(indexed, key=lambda item: item[1].reward)[0]


def _task_frontier(population: Population, *, role: str) -> list[PromptCandidate]:
    """Return candidates that are best for at least one task."""
    task_ids = _population_task_ids(population)
    frontier: list[PromptCandidate] = []
    frontier_ids: set[int] = set()
    for index, task_id in enumerate(task_ids):
        contenders = [
            candidate
            for candidate in population.members
            if index < len(candidate.breakdowns)
        ]
        if not contenders:
            continue
        best = max(contenders, key=lambda candidate: _task_objective(candidate, index, role))
        best.frontier_task_ids.append(task_id)
        if id(best) not in frontier_ids:
            frontier.append(best)
            frontier_ids.add(id(best))
    return frontier


def _population_task_ids(population: Population) -> list[str]:
    for candidate in population.members:
        if candidate.task_ids:
            return list(candidate.task_ids)
    max_breakdowns = max((len(candidate.breakdowns) for candidate in population.members), default=0)
    return [f"task:{index}" for index in range(max_breakdowns)]


def _task_objective(candidate: PromptCandidate, index: int, role: str) -> tuple[float, ...]:
    breakdown = candidate.breakdowns[index]
    if role == "distinguisher":
        return (
            breakdown.benchmark_score,
            -breakdown.regression_penalty,
            breakdown.benchmark_delta,
            breakdown.type1_fixed_count,
            -breakdown.type2_count,
            -breakdown.type1_count,
            candidate.reward,
        )
    role_reward = breakdown.r_generator
    return (
        breakdown.benchmark_score,
        -breakdown.regression_penalty,
        breakdown.benchmark_delta,
        role_reward,
        candidate.reward,
    )


def _rank_frontier(candidates: list[PromptCandidate]) -> list[PromptCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            len(candidate.frontier_task_ids),
            candidate.reward,
            _hash_prompt(candidate.prompt),
        ),
        reverse=True,
    )


def _write_population_snapshot(
    run_dir: Path,
    *,
    epoch: int,
    role: str,
    population: Population,
) -> None:
    snapshots_dir = run_dir / "prompt_populations"
    snapshots_dir.mkdir(exist_ok=True)
    rows = []
    ranked = sorted(
        population.members,
        key=lambda candidate: candidate.reward,
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, 1):
        breakdowns = candidate.breakdowns or []
        rows.append(
            {
                "epoch": epoch,
                "role": role,
                "rank": rank,
                "prompt_hash": _hash_prompt(candidate.prompt),
                "reward": candidate.reward,
                "avg_benchmark": (
                    mean(b.benchmark_score for b in breakdowns) if breakdowns else None
                ),
                "avg_type1_count": (
                    mean(b.type1_count for b in breakdowns) if breakdowns else None
                ),
                "avg_type1_fixed_count": (
                    mean(b.type1_fixed_count for b in breakdowns) if breakdowns else None
                ),
                "avg_type2_count": (
                    mean(b.type2_count for b in breakdowns) if breakdowns else None
                ),
                "avg_useful_clarification_rate": (
                    mean(b.useful_clarification_rate for b in breakdowns)
                    if breakdowns
                    else None
                ),
                "avg_benchmark_delta": (
                    mean(b.benchmark_delta for b in breakdowns) if breakdowns else None
                ),
                "avg_regression_penalty": (
                    mean(b.regression_penalty for b in breakdowns) if breakdowns else None
                ),
                "task_ids": candidate.task_ids,
                "frontier_task_ids": candidate.frontier_task_ids,
                "anchor": candidate.anchor,
                "failure_summaries": candidate.failure_summaries[:8],
                "prompt": candidate.prompt,
            }
        )
    path = snapshots_dir / f"{role}_epoch_{epoch:03d}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


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
    failure_summaries: list[str] = []
    rewards: list[float] = []
    task_ids: list[str] = []
    for i, task in enumerate(tasks, 1):
        task_ids.append(task.id)
        generator_prompt = candidate.prompt if role == "generator" else opponent_prompt
        distinguisher_prompt = candidate.prompt if role == "distinguisher" else opponent_prompt

        generator = Generator(llm, prompt=generator_prompt)
        distinguisher = Distinguisher(llm, prompt=distinguisher_prompt)
        user = user_factory(task)

        try:
            trace = run_episode(task, generator, distinguisher, user, max_turns=config.max_turns)
            trace.generator_prompt_hash = _hash_prompt(generator_prompt)
            trace.distinguisher_prompt_hash = _hash_prompt(distinguisher_prompt)
            score_result = _score_trace_detailed(task, trace, coder)
            benchmark = score_result.pass_rate
            task_failure_summaries = _format_failure_summaries(task, score_result)
            failure_summaries.extend(task_failure_summaries)
            failure_context = _format_failure_context(task, trace, score_result)
            if failure_context is not None:
                failure_summaries.append(failure_context)
            trace.benchmark_score = benchmark
            breakdown = compute_reward_breakdown(
                trace,
                task.prompt,
                benchmark_score=benchmark,
                weights=weights,
                baseline_score=baselines.get(task.id),
            )
            trace.rewards = breakdown
            error_type = None
            error_message = None
        except BudgetExceededError:
            raise
        except Exception as exc:
            benchmark = 0.0
            breakdown = RewardBreakdown(
                benchmark_score=benchmark,
                r_generator=-1.0,
                r_distinguisher=-1.0,
            )
            error_type = type(exc).__name__
            error_message = str(exc)
            task_failure_summaries = [f"{task.id}: scoring error: {error_message}"]
            trace = None
            log.warning(
                "      task %d/%d %s failed for %s candidate %s: %s: %s",
                i,
                len(tasks),
                task.id,
                role,
                _hash_prompt(candidate.prompt),
                error_type,
                error_message,
            )

        breakdowns.append(breakdown)
        rewards.append(breakdown.r_generator if role == "generator" else breakdown.r_distinguisher)
        log.info(
            "      task %d/%d %s: turns=%d benchmark=%.3f reward=%.3f%s",
            i,
            len(tasks),
            task.id,
            len(trace.turns) if trace is not None else 0,
            benchmark,
            rewards[-1],
            _calls_so_far(llm),
        )
        if run_dir is not None:
            if trace is not None:
                write_trace(run_dir, trace)
            metrics = _trace_metrics(
                epoch=epoch,
                role=role,
                task_id=task.id,
                prompt=candidate.prompt,
                reward=rewards[-1],
                benchmark_score=benchmark,
                llm=llm,
            )
            if error_type is not None:
                metrics["error_type"] = error_type
                metrics["error_message"] = error_message[:500] if error_message else ""
            if task_failure_summaries:
                metrics["failure_summaries"] = task_failure_summaries[:3]
            write_metrics(run_dir, metrics)

    candidate.reward = mean(rewards) if rewards else 0.0
    candidate.breakdowns = breakdowns
    candidate.failure_summaries = failure_summaries[:8]
    candidate.task_ids = task_ids
    return candidate


def _score_trace(task: Task, trace: Trace, coder: Coder) -> float:
    return _score_trace_detailed(task, trace, coder).pass_rate


def _score_trace_detailed(task: Task, trace: Trace, coder: Coder) -> ScoreResult:
    if trace.final_spec is None:
        return ScoreResult(0, 0, 0.0, errors=["no final spec"])
    code = coder.from_spec(trace.final_spec, task.prompt)
    return score_detailed(task, code)


def _format_failure_summaries(task: Task, result: ScoreResult) -> list[str]:
    if result.pass_count == result.total_count:
        return []
    summaries: list[str] = []
    for example in result.failure_examples[:3]:
        actual = example.actual_repr if example.actual_repr is not None else example.error
        hint = _failure_hint(example)
        hint_suffix = f"; hint={hint}" if hint else ""
        summaries.append(
            f"{task.id}: plus score {result.pass_count}/{result.total_count}; "
            f"input={example.input_repr}; expected={example.expected_repr}; "
            f"actual={actual}{hint_suffix}"
        )
    if summaries:
        return summaries
    if result.errors:
        return [f"{task.id}: scoring error: {result.errors[0]}"]
    return [f"{task.id}: plus score {result.pass_count}/{result.total_count}"]


def _format_failure_context(
    task: Task,
    trace: Trace,
    result: ScoreResult,
) -> str | None:
    if result.pass_count == result.total_count or trace.final_spec is None:
        return None
    spec = trace.final_spec
    issue_summaries = [
        (
            f"{issue.route}/{issue.kind} at {issue.location}: "
            f"{_truncate(issue.description, 140)}"
        )
        for turn in trace.turns
        for issue in turn.issues
    ][:4]
    spec_bits = {
        "goal": _truncate(spec.goal, 180),
        "preconditions": [_truncate(item, 140) for item in spec.preconditions[:3]],
        "postconditions": [_truncate(item, 140) for item in spec.postconditions[:3]],
        "edge_cases": [_truncate(item, 140) for item in spec.edge_cases[:4]],
        "acceptance_criteria": [
            _truncate(item, 140) for item in spec.acceptance_criteria[:4]
        ],
    }
    return (
        f"{task.id}: failure context; "
        f"task_prompt={_truncate(task.prompt, 240)!r}; "
        f"final_spec={json.dumps(spec_bits, sort_keys=True)}; "
        f"d_issues={json.dumps(issue_summaries)}"
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _failure_hint(example: FailureExample) -> str | None:
    expected = _literal_eval_repr(example.expected_repr)
    actual = _literal_eval_repr(example.actual_repr)
    if not isinstance(expected, str) or not isinstance(actual, str):
        return None

    hints: list[str] = []
    if expected and _is_subsequence(expected, actual) and expected != actual:
        hints.append("expected is a filtered subsequence of actual")
    if expected and expected.isalpha() and expected.islower():
        hints.append("expected contains only lowercase alphabetic characters")
    extra_chars = {char for char in actual if char not in expected}
    if any(not char.isalnum() for char in extra_chars):
        hints.append("actual preserved punctuation/symbols absent from expected")
    if any(char.isupper() for char in extra_chars):
        hints.append("actual preserved uppercase letters absent from expected")
    if not hints:
        return None
    return "; ".join(hints)


def _literal_eval_repr(value: str | None) -> object | None:
    if value is None:
        return None
    try:
        parsed: object = ast.literal_eval(value)
        return parsed
    except (SyntaxError, ValueError):
        return None


def _is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


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


def _summarize_feedback(
    candidate: PromptCandidate,
    role: str,
    peer_elites: list[PromptCandidate] | None = None,
) -> str:
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
    avg_benchmark_delta = mean(b.benchmark_delta for b in candidate.breakdowns)
    avg_regression_penalty = mean(b.regression_penalty for b in candidate.breakdowns)
    failure_feedback = ""
    if candidate.failure_summaries:
        examples = "\n".join(
            f"- {summary}" for summary in candidate.failure_summaries[:5]
        )
        failure_feedback = (
            " Concrete failed hidden-test examples to learn from:\n"
            f"{examples}\n"
            "Turn these into reusable semantic rules in the next prompt; "
            "do not merely mention the individual inputs."
        )
    merge_feedback = _summarize_peer_elites(candidate, peer_elites)

    if role == "generator":
        return (
            f"Generator results over {n} tasks. "
            f"avg benchmark={avg_benchmark:.3f} (higher is better). "
            f"avg benchmark delta vs direct baseline={avg_benchmark_delta:.3f}; "
            f"avg regression penalty={avg_regression_penalty:.3f} "
            f"(keep this at 0 by not losing hidden tests direct code passed). "
            f"avg type-1 issues D raised against your specs={avg_type1:.2f} "
            f"(lower is better — these are gaps you should have caught). "
            f"avg spec complexity penalty={avg_spec_penalty:.3f} "
            f"(avoid empty / thin specs). Improve by producing specs "
            f"concrete enough that D has fewer gaps to flag, without "
            f"dropping benchmark score.{failure_feedback}{merge_feedback}"
        )
    return (
        f"Distinguisher results over {n} tasks. "
        f"avg benchmark={avg_benchmark:.3f}. "
        f"avg benchmark delta vs direct baseline={avg_benchmark_delta:.3f}; "
        f"avg regression penalty={avg_regression_penalty:.3f} "
        f"(keep this at 0 by not pushing specs that lose hidden tests). "
        f"avg type-1 issues you raised={avg_type1:.2f}; "
        f"of those, avg actually fixed next turn={avg_type1_fixed:.2f} "
        f"(this is your accepted-reject rate — higher is better). "
        f"avg type-2 questions={avg_type2:.2f}; "
        f"avg dismissed by user={avg_type2_dismissed:.2f} "
        f"(lower is better). useful clarification rate={avg_useful_rate:.3f} "
        f"(positive means your questions improved the spec). "
        f"Improve by raising issues G will accept and asking only "
        f"genuinely useful user-routed questions (cap is 5/episode)."
        f"{failure_feedback}{merge_feedback}"
    )


def _summarize_peer_elites(
    candidate: PromptCandidate,
    peer_elites: list[PromptCandidate] | None,
) -> str:
    if not peer_elites:
        return ""
    peers = [
        peer
        for peer in peer_elites
        if peer is not candidate and (peer.frontier_task_ids or peer.failure_summaries)
    ][:3]
    if not peers:
        return ""
    lines: list[str] = []
    for peer in peers:
        frontier = ", ".join(peer.frontier_task_ids[:5]) or "no specific frontier task"
        failures = "; ".join(peer.failure_summaries[:2]) or "no failure examples"
        lines.append(
            f"- peer {_hash_prompt(peer.prompt)} survived on {frontier}; "
            f"lessons/failures: {failures}"
        )
    return (
        " Complementary elite prompts also survived this epoch:\n"
        + "\n".join(lines)
        + "\nMerge their reusable lessons when they apply; do not copy "
        "task IDs or individual hidden-test inputs as lookup rules."
    )


def _trace_metrics(
    *,
    epoch: int,
    role: str,
    task_id: str,
    prompt: str,
    reward: float,
    benchmark_score: float,
    llm: LLMClient | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "epoch": epoch,
        "role": role,
        "task_id": task_id,
        "prompt_hash": _hash_prompt(prompt),
        "reward": reward,
        "benchmark_score": benchmark_score,
    }
    if llm is not None:
        metrics["llm_structured_fallback_count"] = getattr(
            llm,
            "structured_fallback_count",
            0,
        )
    return metrics
