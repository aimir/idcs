from __future__ import annotations

from pydantic import BaseModel

from idcs.optimizer.gepa_adapter import (
    DISTINGUISHER_COMPONENT,
    GENERATOR_COMPONENT,
    IDCSGepaAdapter,
    compute_direct_baselines,
    seed_candidate,
)
from idcs.schemas import IssueList, Spec, Task, Test
from tests.fakes import FakeLLM


def _task() -> Task:
    return Task(
        id="seed/gepa-add",
        prompt="Write add(a, b) returning the arithmetic sum of two integers.",
        entry_point="add",
        tests=[
            Test(id="seed/gepa-add/t0", code="assert add(1, 2) == 3"),
            Test(id="seed/gepa-add/t1", code="assert add(-2, 5) == 3"),
        ],
    )


def _spec() -> Spec:
    return Spec(
        goal="Return the arithmetic sum of a and b.",
        inputs=[
            {"name": "a", "type": "int", "description": "First addend."},
            {"name": "b", "type": "int", "description": "Second addend."},
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


def test_adapter_evaluates_candidate_without_gepa_dependency() -> None:
    task = _task()

    def typed_responder(
        system: str,
        user: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        assert system in {"generator candidate", "distinguisher candidate"}
        if output_type is Spec:
            assert "MODE: draft" in user
            return _spec()
        if output_type is IssueList:
            assert "PROPOSED SPEC" in user
            return IssueList(issues=[])
        raise AssertionError(f"unexpected output type: {output_type.__name__}")

    llm = FakeLLM(
        typed_responder=typed_responder,
        text_responses=["def add(a, b):\n    return a + b\n"],
    )
    adapter = IDCSGepaAdapter(
        llm=llm,
        generator_prompt="generator seed",
        distinguisher_prompt="distinguisher seed",
        baseline_scores={task.id: 1.0},
        max_turns=2,
    )

    batch = adapter.evaluate(
        [task],
        {
            GENERATOR_COMPONENT: "generator candidate",
            DISTINGUISHER_COMPONENT: "distinguisher candidate",
        },
        capture_traces=True,
    )

    assert batch.scores == [1.0]
    assert batch.outputs[0].benchmark_score == 1.0
    assert batch.outputs[0].benchmark_delta == 0.0
    assert batch.outputs[0].pass_count == 2
    assert batch.trajectories is not None
    assert batch.trajectories[0].trace is not None


def test_reflective_dataset_contains_component_specific_feedback() -> None:
    task = _task()
    llm = FakeLLM(
        typed_responses=[_spec(), IssueList(issues=[])],
        text_responses=["def add(a, b):\n    return a + b\n"],
    )
    adapter = IDCSGepaAdapter(
        llm=llm,
        generator_prompt="generator seed",
        distinguisher_prompt="distinguisher seed",
        baseline_scores={task.id: 0.5},
        max_turns=1,
    )
    candidate = seed_candidate(
        generator_prompt="generator candidate",
        distinguisher_prompt="distinguisher candidate",
    )
    batch = adapter.evaluate([task], candidate, capture_traces=True)

    dataset = adapter.make_reflective_dataset(
        candidate,
        batch,
        [GENERATOR_COMPONENT, DISTINGUISHER_COMPONENT],
    )

    generator_record = dataset[GENERATOR_COMPONENT][0]
    assert generator_record["Task"]["prompt"] == task.prompt
    assert generator_record["Score"]["plus_tests"] == "2/2"
    assert generator_record["Score"]["benchmark_delta_vs_direct"] == 0.5
    assert "spec generator" in generator_record["Component-specific guidance"]
    assert "Do not copy task ids" in generator_record["Instruction"]

    distinguisher_record = dataset[DISTINGUISHER_COMPONENT][0]
    assert "distinguisher" in distinguisher_record["Component-specific guidance"]


def test_adapter_uses_role_specific_llms_for_evaluation() -> None:
    task = _task()
    fallback_llm = FakeLLM()
    generator_llm = FakeLLM(typed_responses=[_spec()])
    distinguisher_llm = FakeLLM(typed_responses=[IssueList(issues=[])])
    coder_llm = FakeLLM(text_responses=["def add(a, b):\n    return a + b\n"])
    adapter = IDCSGepaAdapter(
        llm=fallback_llm,
        generator_llm=generator_llm,
        distinguisher_llm=distinguisher_llm,
        coder_llm=coder_llm,
        generator_prompt="generator seed",
        distinguisher_prompt="distinguisher seed",
        max_turns=1,
    )
    candidate = seed_candidate(
        generator_prompt="generator candidate",
        distinguisher_prompt="distinguisher candidate",
    )

    batch = adapter.evaluate([task], candidate, capture_traces=True)

    assert batch.scores == [1.0]
    assert len(generator_llm.typed_calls) == 1
    assert generator_llm.typed_calls[0][0] == "generator candidate"
    assert len(distinguisher_llm.typed_calls) == 1
    assert distinguisher_llm.typed_calls[0][0] == "distinguisher candidate"
    assert len(coder_llm.text_calls) == 1
    assert fallback_llm.typed_calls == []
    assert fallback_llm.text_calls == []


def test_compute_direct_baselines_scores_prompt_path() -> None:
    task = _task()
    llm = FakeLLM(text_responses=["def add(a, b):\n    return a + b\n"])

    baselines = compute_direct_baselines([task], llm)

    assert baselines == {task.id: 1.0}
    assert "MODE: from_prompt" in llm.text_calls[0][1]


def test_adapter_proposes_new_texts_with_existing_mutator() -> None:
    fallback_llm = FakeLLM()
    mutator_llm = FakeLLM(
        typed_responder=lambda system, user, output_type: output_type(
            prompts=["# Generator improved\n\nWrite edge cases explicitly."]
        )
    )
    adapter = IDCSGepaAdapter(
        llm=fallback_llm,
        mutator_llm=mutator_llm,
        generator_prompt="generator seed",
        distinguisher_prompt="distinguisher seed",
    )

    proposed = adapter.propose_new_texts(
        {GENERATOR_COMPONENT: "generator current"},
        {
            GENERATOR_COMPONENT: [
                {
                    "Score": {"benchmark_score": 0.25},
                    "Failed hidden-test feedback": ["expected lowercase only"],
                }
            ]
        },
        [GENERATOR_COMPONENT],
    )

    assert proposed == {
        GENERATOR_COMPONENT: "# Generator improved\n\nWrite edge cases explicitly."
    }
    assert len(mutator_llm.typed_calls) == 1
    assert "ROLE: generator" in mutator_llm.typed_calls[0][1]
    assert "GEPA selected generator_prompt" in mutator_llm.typed_calls[0][1]
    assert "High-signal summary" in mutator_llm.typed_calls[0][1]
    assert "expected lowercase only" in mutator_llm.typed_calls[0][1]
    assert fallback_llm.typed_calls == []
