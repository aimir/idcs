# Design

## Core abstractions

```python
class Task:           # one benchmark item: NL prompt + hidden test suite
    id: str
    prompt: str
    tests: list[Test] # used only to score the final spec; never seen by G/D

class Spec:           # structured artifact G produces
    goal: str
    inputs: list[Field]
    outputs: list[Field]
    preconditions: list[str]
    postconditions: list[str]
    invariants: list[str]
    edge_cases: list[str]
    acceptance_criteria: list[str]

class Issue:
    kind: Literal[
        "gap", "ambiguity", "contradiction",
        "over_constraint", "underconstraint", "implicit_assumption",
    ]
    route: Literal["generator", "user"]   # type-1 vs type-2
    location: SpecPath                     # which field
    description: str
    suggested_question: str | None         # only if route=user

class Trace:          # full episode record
    task_id: str
    turns: list[Turn]                      # (spec_v, issues, routing_decisions, user_answers)
    final_spec: Spec
    benchmark_score: float
    rewards: RewardBreakdown
```

## Pipeline (one episode)

```
input ──► G ──► spec_0
                │
                ▼
              D ──► issues_0
                │
        ┌───────┴───────┐
        ▼               ▼
  type-1 issues   type-2 issues
        │               │
        │           user_proxy.answer(question)
        │               │
        └──────┬────────┘
               ▼
              G(spec_0, issues, answers) ──► spec_1 ──► ... (until D returns ∅ or max_turns)
                                                            │
                                                            ▼
                                              coder(spec) ──► external grader ──► benchmark_score
```

`user_proxy` during training is an oracle LLM with access to the gold
reference; during real use it is the human. **Important**: the oracle must
give minimal answers (only what the question asks, with noise), otherwise D
learns to extract gold-level information for free.

## Reward structure

Benchmark dominant; per-role terms with shared anti-regression and excess-clarification penalties (`src/idcs/rewards.py`):

```
R_G = α · benchmark_score
    − β · count(type1_issues_raised_against_G)
    − γ · spec_complexity_penalty                  # discourages echo-the-input collapse
    − ρ · regression_penalty                       # vs direct baseline (see below)

R_D = α · benchmark_score
    + β · count(type1_issues_G_actually_fixed)     # only credit accepted rejects
    + δ · useful_clarification_rate                # see below
    − ε · count(type2_issues_user_dismissed)
    − ζ · max(0, type2_count − max_type2_per_episode)   # cap on type-2 spam
    − ρ · regression_penalty

useful_clarification_rate =
    (benchmark_score − direct_baseline) / max(1, number of type-2 questions asked)

regression_penalty = max(0, direct_baseline − benchmark_score)   # only positive when spec-guided loses
```

The **clarification rate** is computed cheaply from the per-task direct baseline rather than via counterfactual reruns — coevolution iterations need this signal hundreds of times per epoch. The **regression penalty** symmetrically penalizes both roles when a spec-guided run loses hidden tests relative to the direct baseline, so coevolution can't trade one rescued task for broad partial regressions elsewhere. The **type-2 cap** penalizes on the cap rather than the average, so a 6th question can't be made profitable by being only slightly useful.

**Suggested scale** (current defaults): `α=1.0`, `β=0.1`, `γ=0.05`, `δ=0.1`, `ε=0.05`, `ζ=1.0`, `ρ=2.0`. The regression penalty is intentionally aggressive: a −0.1 hidden-test delta erases a full task pass.

## Optimization

Prompts, not weights:

- **Population-based prompt search** (GEPA / PromptBreeder style). Maintain
  populations `P_G` (~16) and `P_D` (~16).
- **Co-evolution**: alternating epochs. In a G-epoch, sample D from `P_D`,
  mutate G prompts, keep top-k by `R_G` on a training task batch. Then swap.
- **Mutator** is itself an LLM call: given a prompt, a batch of failure
  traces, and the reward breakdown, propose 3 edited prompts. Looking at
  actual traces beats blind mutation.
- **Diversity guard**: enforce pairwise edit distance within each population
  to prevent collapse onto a single point.

## Code layout

```
idcs/
  pyproject.toml
  src/idcs/
    schemas.py              # pydantic models above
    llm.py                  # OpenAI-SDK wrapper; OpenRouter / OpenAI / local codex backends
    generator.py            # G.draft(task) and G.revise(spec, issues, answers)
    distinguisher.py        # D.critique(task, spec) -> list[Issue]
    user_proxy.py           # Oracle / human dispatcher
    orchestrator.py         # run_episode(task, G, D, user) -> Trace
    coder.py                # spec -> code (frozen prompt, not optimized)
    benchmark/
      tasks.py              # adapter for the external benchmark library
      scoring.py            # thin wrapper around the library's grader call
    rewards.py              # compute R_G, R_D from Trace
    optimizer/
      population.py         # candidate populations with diversity guard
      mutate.py             # LLM-driven prompt mutation w/ plain-text fallback
      coevolve.py           # outer training loop, task-Pareto elite selection,
                            # anchor protection, held-out val split
    telemetry.py            # structured logging of every turn
  prompts/
    generator_v0.md
    distinguisher_v0.md
    mutator.md
    coder.md
  data/
    seed_tasks/             # ~50 hand-written (input, gold_spec, plausible_gaps)
    benchmarks/
  experiments/
    configs/*.yaml
    runs/<timestamp>/       # traces, populations, metrics
  scripts/
    cold_start.py           # validate G and D on seed tasks
    train.py                # coevolve
    eval.py                 # frozen-prompt eval on held-out benchmark
    inspect.py              # CLI to step through a trace
  tests/
```

Two modules deserve attention and are easy to underestimate:

- `optimizer/coevolve.py`: task-Pareto elite selection, anchor-protected
  base prompt, held-out validation split, and rich failure-context
  feedback into the next mutation prompt. The pieces that prevent
  coevolution from collapsing onto a single trick.
- `telemetry.py`: complete structured traces with prompt hashes, config
  snapshot, and per-turn issue routing. You cannot debug coevolution
  without them.
