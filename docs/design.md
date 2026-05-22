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
    kind: Literal["gap", "ambiguity", "contradiction", "over_constraint"]
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
                                                  spec→code→tests ──► benchmark_score
```

`user_proxy` during training is an oracle LLM with access to the gold
reference; during real use it is the human. **Important**: the oracle must
give minimal answers (only what the question asks, with noise), otherwise D
learns to extract gold-level information for free.

## Reward structure

Three terms per side, with the benchmark dominant:

```
R_G = α · benchmark_score
    − β · count(type1_issues_raised_against_G)
    − γ · spec_complexity_penalty       # discourages echo-the-input collapse

R_D = α · benchmark_score
    + β · count(type1_issues_G_actually_fixed)    # only credit accepted rejects
    + δ · useful_clarification_rate                # see below
    − ε · count(type2_issues_user_dismissed)

useful_clarification_rate =
    Σ (Δ benchmark_score attributable to question q) / (number of type-2 questions asked)
```

**Attribution** is the tricky bit: run a counterfactual episode where the
type-2 issue is dropped and measure the benchmark delta. Expensive, but only
needed offline during training and cacheable per (task, issue).

**Suggested scale**: `α` should be 5–10× the other coefficients. The
`spec_complexity_penalty` only kicks in at the extremes (penalize if spec
length < input length, or embedding distance to input below a threshold).

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
    llm.py                  # Anthropic SDK wrapper, prompt caching enabled
    generator.py            # G.draft(task) and G.revise(spec, issues, answers)
    distinguisher.py        # D.critique(task, spec) -> list[Issue]
    user_proxy.py           # Oracle / human dispatcher
    orchestrator.py         # run_episode(task, G, D, user) -> Trace
    coder.py                # spec -> code (frozen prompt, not optimized)
    benchmark/
      tasks.py              # load HumanEval+/MBPP+/custom
      runner.py             # sandboxed test execution
      score.py
    rewards.py              # compute R_G, R_D from Trace
    optimizer/
      population.py
      mutate.py             # LLM-driven prompt mutation
      coevolve.py           # outer training loop
      attribution.py        # counterfactual rerun for type-2 credit
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

- `attribution.py`: counterfactual reruns for type-2 credit assignment.
- `telemetry.py`: complete structured traces. You cannot debug coevolution
  without them.
