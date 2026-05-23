"""Drive one episode of the G → D → route → G loop."""

from __future__ import annotations

from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.schemas import Task, Trace, Turn
from idcs.user_proxy import UserProxy


def run_episode(
    task: Task,
    generator: Generator,
    distinguisher: Distinguisher,
    user: UserProxy,
    *,
    max_turns: int = 5,
) -> Trace:
    """Run G → D → route until D approves the spec or ``max_turns`` is reached.

    Each ``Turn`` in the trace records the spec D evaluated, the issues D
    raised, and any user answers we collected. The loop exits as soon as D
    returns no issues; ``final_spec`` is the last spec produced — by D's
    most recent ``revise`` if it ran, otherwise the initial ``draft``.
    """
    spec = generator.draft(task.prompt)
    turns: list[Turn] = []

    for _ in range(max_turns):
        issues = distinguisher.critique(task.prompt, spec)
        answers: dict[str, str] = {}
        for issue in issues:
            if issue.route == "user" and issue.suggested_question:
                response = user.answer(issue.location, issue.suggested_question)
                if response is not None:
                    answers[issue.location] = response

        turns.append(Turn(spec=spec, issues=issues, user_answers=answers))

        if not issues:
            break

        spec = generator.revise(task.prompt, spec, issues, answers)

    return Trace(task_id=task.id, turns=turns, final_spec=spec)
