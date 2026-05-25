from __future__ import annotations

from pydantic import BaseModel

from idcs.optimizer.coevolve import CoevolveConfig, coevolve
from idcs.rewards import RewardWeights
from idcs.schemas import Issue, IssueList, Spec, Task, Test
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


def test_coevolve_runs_with_fake_llms_end_to_end() -> None:
    task = _task()
    users: list[FakeUserProxy] = []
    state = {"issue_pending": False, "mutation_index": 0}

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
            telemetry=False,
            seed=7,
        ),
        mutator_llm=mutator_llm,
    )

    assert result.run_dir is None
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
