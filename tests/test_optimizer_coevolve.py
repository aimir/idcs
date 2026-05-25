from __future__ import annotations

import json
import random

from pydantic import BaseModel

from idcs.benchmark.scoring import FailureExample, ScoreResult
from idcs.optimizer.coevolve import (
    CoevolveConfig,
    _evaluate_candidate,
    _format_failure_context,
    _format_failure_summaries,
    _select_elites,
    _summarize_feedback,
    coevolve,
)
from idcs.optimizer.population import Population, PromptCandidate
from idcs.rewards import RewardWeights
from idcs.schemas import Issue, IssueList, RewardBreakdown, Spec, Task, Test, Trace, Turn
from tests.fakes import FakeLLM, FakeUserProxy


def _task() -> Task:
    return Task(
        id="seed/coevolve-add",
        prompt="Write add(a, b) returning the arithmetic sum of two integers.",
        entry_point="add",
        tests=[
            Test(id="seed/coevolve-add/t0", code="assert add(1, 2) == 3"),
            Test(id="seed/coevolve-add/t1", code="assert add(-2, 5) == 3"),
        ],
    )


def _spec(goal: str) -> Spec:
    return Spec(
        goal=goal,
        inputs=[
            {
                "name": "a",
                "type": "int",
                "description": "First addend.",
            },
            {
                "name": "b",
                "type": "int",
                "description": "Second addend.",
            },
        ],
        outputs=[
            {
                "name": "result",
                "type": "int",
                "description": "The arithmetic sum a + b.",
            }
        ],
        acceptance_criteria=["add(1, 2) returns 3", "add(-2, 5) returns 3"],
    )


def test_coevolve_runs_with_fake_llms_end_to_end(tmp_path, monkeypatch) -> None:
    task = _task()
    users: list[FakeUserProxy] = []
    state = {"issue_pending": False, "mutation_index": 0}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr("idcs.optimizer.coevolve.create_run_dir", lambda: run_dir)

    def text_responder(system: str, user: str) -> str:
        assert system
        assert "MODE: from_" in user
        return "def add(a, b):\n    return a + b\n"

    def main_typed_responder(
        system: str,
        user: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        assert system
        if output_type is Spec:
            if "MODE: draft" in user:
                state["issue_pending"] = True
                return _spec("Draft a deterministic add implementation.")
            assert "MODE: revise" in user
            assert "Use ordinary Python integer addition." in user
            return _spec("Revised add implementation with clarified integer semantics.")

        if output_type is IssueList:
            if state["issue_pending"]:
                state["issue_pending"] = False
                return IssueList(
                    issues=[
                        Issue(
                            kind="ambiguity",
                            route="user",
                            location="preconditions[0]",
                            description="Integer semantics are not explicit.",
                            suggested_question="Should add use ordinary Python integer addition?",
                        )
                    ]
                )
            return IssueList(issues=[])

        raise AssertionError(f"unexpected main output type: {output_type.__name__}")

    def mutator_typed_responder(
        system: str,
        user: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        assert system
        assert output_type.__name__ == "MutationBatch"
        role = "generator" if "ROLE: generator" in user else "distinguisher"
        state["mutation_index"] += 1
        return output_type(
            prompts=[f"{role} mutation {state['mutation_index']}: keep behavior deterministic"]
        )

    def user_factory(task: Task) -> FakeUserProxy:
        user = FakeUserProxy(
            answers={"preconditions[0]": "Use ordinary Python integer addition."}
        )
        users.append(user)
        return user

    main_llm = FakeLLM(
        text_responder=text_responder,
        typed_responder=main_typed_responder,
    )
    mutator_llm = FakeLLM(typed_responder=mutator_typed_responder)

    result = coevolve(
        [task],
        main_llm,
        generator_prompt="generator base prompt",
        distinguisher_prompt="distinguisher base prompt",
        user_factory=user_factory,
        weights=RewardWeights(min_spec_ratio=0.0),
        config=CoevolveConfig(
            population_size=2,
            elite_size=1,
            epochs=1,
            max_turns=2,
            telemetry=True,
            seed=7,
        ),
        mutator_llm=mutator_llm,
    )

    assert result.run_dir == run_dir
    assert len(result.generator.members) == 2
    assert len(result.distinguisher.members) == 2
    assert result.generator.best().reward == 1.0
    assert result.distinguisher.best().reward == 1.0
    assert any("generator mutation" in c.prompt for c in result.generator.members)
    assert any("distinguisher mutation" in c.prompt for c in result.distinguisher.members)

    assert len(users) == 4
    assert all(
        user.calls == [
            ("preconditions[0]", "Should add use ordinary Python integer addition?")
        ]
        for user in users
    )

    main_typed_names = [call[2].__name__ for call in main_llm.typed_calls]
    assert main_typed_names.count("Spec") == 8
    assert main_typed_names.count("IssueList") == 8
    assert "MutationBatch" not in main_typed_names

    assert len(main_llm.text_calls) == 5
    assert "MODE: from_prompt" in main_llm.text_calls[0][1]
    assert all("MODE: from_spec" in call[1] for call in main_llm.text_calls[1:])
    assert [call[2].__name__ for call in mutator_llm.typed_calls] == [
        "MutationBatch",
        "MutationBatch",
        "MutationBatch",
        "MutationBatch",
    ]

    generator_snapshot = json.loads(
        (run_dir / "prompt_populations" / "generator_epoch_001.json").read_text(
            encoding="utf-8"
        )
    )
    distinguisher_snapshot = json.loads(
        (run_dir / "prompt_populations" / "distinguisher_epoch_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(generator_snapshot) == 2
    assert len(distinguisher_snapshot) == 2
    assert generator_snapshot[0]["rank"] == 1
    assert generator_snapshot[0]["reward"] == 1.0
    assert generator_snapshot[0]["prompt"]
    assert distinguisher_snapshot[0]["rank"] == 1
    assert distinguisher_snapshot[0]["reward"] == 1.0
    assert distinguisher_snapshot[0]["prompt"]


def test_candidate_evaluation_records_failure_without_crashing(tmp_path) -> None:
    task = _task()

    def failing_typed_responder(
        system: str,
        user: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        del system, user, output_type
        raise RuntimeError("malformed structured output")

    llm = FakeLLM(typed_responder=failing_typed_responder)
    candidate = PromptCandidate(prompt="generator prompt")

    result = _evaluate_candidate(
        role="generator",
        candidate=candidate,
        opponent_population=Population([PromptCandidate(prompt="distinguisher prompt")]),
        tasks=[task],
        llm=llm,
        user_factory=lambda task: FakeUserProxy(),
        weights=RewardWeights(),
        config=CoevolveConfig(max_turns=1),
        rng=random.Random(7),
        run_dir=tmp_path,
        epoch=1,
        baselines={},
    )

    assert result.reward == -1.0
    assert result.breakdowns[0].benchmark_score == 0.0

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics == [
        {
            "epoch": 1,
            "role": "generator",
            "task_id": "seed/coevolve-add",
            "prompt_hash": "11167aa98036",
            "reward": -1.0,
            "benchmark_score": 0.0,
            "llm_structured_fallback_count": 0,
            "error_type": "RuntimeError",
            "error_message": "malformed structured output",
            "failure_summaries": [
                "seed/coevolve-add: scoring error: malformed structured output"
            ],
        }
    ]


def test_mutation_feedback_includes_delta_and_regression_terms() -> None:
    candidate = PromptCandidate(
        prompt="generator prompt",
        breakdowns=[
            RewardBreakdown(
                benchmark_score=0.4,
                benchmark_delta=0.2,
                regression_penalty=0.0,
            ),
            RewardBreakdown(
                benchmark_score=0.2,
                benchmark_delta=-0.1,
                regression_penalty=0.1,
            ),
        ],
        failure_summaries=[
            "Mbpp/459: plus score 12/50; input=['AbC']; expected='b'; actual='AbC'"
        ],
    )

    feedback = _summarize_feedback(candidate, "generator")

    assert "avg benchmark delta vs direct baseline=0.050" in feedback
    assert "avg regression penalty=0.050" in feedback
    assert "Concrete failed hidden-test examples" in feedback
    assert "Mbpp/459: plus score 12/50" in feedback
    assert "Turn these into reusable semantic rules" in feedback


def test_mutation_feedback_can_include_peer_elite_lessons() -> None:
    candidate = PromptCandidate(
        prompt="generator prompt",
        breakdowns=[RewardBreakdown(benchmark_score=0.5, r_generator=0.5)],
    )
    peer = PromptCandidate(
        prompt="peer prompt",
        reward=0.8,
        frontier_task_ids=["Mbpp/459"],
        failure_summaries=["Mbpp/459: plus score 25/103; expected lowercase only"],
    )

    feedback = _summarize_feedback(candidate, "generator", peer_elites=[candidate, peer])

    assert "Complementary elite prompts also survived" in feedback
    assert "Mbpp/459" in feedback
    assert "Merge their reusable lessons" in feedback


def test_task_pareto_elites_keep_complementary_specialists() -> None:
    candidate_a = PromptCandidate(
        prompt="excellent on task zero",
        reward=0.5,
        task_ids=["task/0", "task/1"],
        breakdowns=[
            RewardBreakdown(benchmark_score=1.0, r_generator=1.0),
            RewardBreakdown(benchmark_score=0.0, r_generator=0.0),
        ],
    )
    candidate_b = PromptCandidate(
        prompt="excellent on task one",
        reward=0.5,
        task_ids=["task/0", "task/1"],
        breakdowns=[
            RewardBreakdown(benchmark_score=0.0, r_generator=0.0),
            RewardBreakdown(benchmark_score=1.0, r_generator=1.0),
        ],
    )
    average_winner = PromptCandidate(
        prompt="smooth average prompt",
        reward=0.7,
        task_ids=["task/0", "task/1"],
        breakdowns=[
            RewardBreakdown(benchmark_score=0.7, r_generator=0.7),
            RewardBreakdown(benchmark_score=0.7, r_generator=0.7),
        ],
    )

    elites = _select_elites(
        Population([average_winner, candidate_a, candidate_b]),
        role="generator",
        config=CoevolveConfig(
            population_size=3,
            elite_size=2,
            elite_selection="task_pareto",
        ),
    )

    assert {candidate.prompt for candidate in elites} == {
        "excellent on task zero",
        "excellent on task one",
    }
    assert candidate_a.frontier_task_ids == ["task/0"]
    assert candidate_b.frontier_task_ids == ["task/1"]
    assert average_winner.frontier_task_ids == []


def test_distinguisher_pareto_tie_break_prefers_non_invasive_critique() -> None:
    noisy = PromptCandidate(
        prompt="noisy distinguisher",
        reward=0.9,
        task_ids=["task/0"],
        breakdowns=[
            RewardBreakdown(
                benchmark_score=1.0,
                r_distinguisher=0.9,
                type1_count=3,
                type2_count=2,
            )
        ],
    )
    quiet = PromptCandidate(
        prompt="quiet distinguisher",
        reward=0.4,
        task_ids=["task/0"],
        breakdowns=[
            RewardBreakdown(
                benchmark_score=1.0,
                r_distinguisher=0.4,
                type1_count=0,
                type2_count=0,
            )
        ],
    )

    elites = _select_elites(
        Population([noisy, quiet]),
        role="distinguisher",
        config=CoevolveConfig(
            population_size=3,
            elite_size=1,
            elite_selection="task_pareto",
            keep_anchor=False,
        ),
    )

    assert [candidate.prompt for candidate in elites] == ["quiet distinguisher"]


def test_mean_elites_keep_average_reward_behavior() -> None:
    pop = Population(
        [
            PromptCandidate(prompt="low", reward=0.1),
            PromptCandidate(prompt="high", reward=0.9),
        ]
    )

    elites = _select_elites(
        pop,
        role="generator",
        config=CoevolveConfig(population_size=3, elite_size=1, elite_selection="mean"),
    )

    assert [candidate.prompt for candidate in elites] == ["high"]


def test_elite_selection_retains_anchor_prompt() -> None:
    anchor = PromptCandidate(prompt="base prompt", reward=0.0, anchor=True)
    specialist_a = PromptCandidate(
        prompt="specialist a",
        reward=0.5,
        task_ids=["task/0", "task/1"],
        breakdowns=[
            RewardBreakdown(benchmark_score=1.0, r_generator=1.0),
            RewardBreakdown(benchmark_score=0.0, r_generator=0.0),
        ],
    )
    specialist_b = PromptCandidate(
        prompt="specialist b",
        reward=0.4,
        task_ids=["task/0", "task/1"],
        breakdowns=[
            RewardBreakdown(benchmark_score=0.0, r_generator=0.0),
            RewardBreakdown(benchmark_score=1.0, r_generator=1.0),
        ],
    )

    elites = _select_elites(
        Population([anchor, specialist_a, specialist_b]),
        role="generator",
        config=CoevolveConfig(
            population_size=3,
            elite_size=1,
            elite_selection="task_pareto",
            keep_anchor=True,
        ),
    )

    assert anchor in elites
    assert len(elites) == 2


def test_failure_summary_adds_string_filter_hint() -> None:
    task = Task(id="Mbpp/459", prompt="", entry_point="remove_uppercase", tests=[])
    result = ScoreResult(
        pass_count=25,
        total_count=103,
        pass_rate=25 / 103,
        failure_examples=[
            FailureExample(
                input_repr="['ThiS%^%!s&a(mY)TesTStR%i*ng']",
                expected_repr="'hisamesting'",
                actual_repr="'hi%^%!s&a(m)est%i*ng'",
            )
        ],
    )

    summary = _format_failure_summaries(task, result)[0]

    assert "expected is a filtered subsequence of actual" in summary
    assert "expected contains only lowercase alphabetic characters" in summary
    assert "actual preserved punctuation/symbols absent from expected" in summary


def test_failure_context_includes_compact_spec_and_d_issues() -> None:
    task = Task(
        id="Mbpp/context",
        prompt="Return the first matching item with exact boundary behavior.",
        entry_point="find_item",
        tests=[],
    )
    issue = Issue(
        kind="underconstraint",
        route="generator",
        location="postconditions[0]",
        description="Tie behavior is not explicit.",
    )
    trace = Trace(
        task_id=task.id,
        turns=[Turn(spec=_spec("Draft spec"), issues=[issue])],
        final_spec=Spec(
            goal="Find a matching item.",
            postconditions=["Return the leftmost matching item when duplicates exist."],
            edge_cases=["Empty input returns None."],
            acceptance_criteria=["Duplicates choose the first occurrence."],
        ),
    )
    result = ScoreResult(pass_count=1, total_count=2, pass_rate=0.5)

    context = _format_failure_context(task, trace, result)

    assert context is not None
    assert "task_prompt='Return the first matching item" in context
    assert "Return the leftmost matching item" in context
    assert "generator/underconstraint" in context
