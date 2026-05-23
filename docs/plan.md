# Implementation plan

Phased so each phase produces something runnable and inspectable before the
next builds on it.

## Phase 0 — Scaffold (1–2 days)

- Repo layout, `pyproject.toml`, ruff/mypy/pytest.
- `schemas.py` — pydantic models from `design.md`.
- `llm.py` — OpenAI SDK wrapper pointed at OpenRouter with caching enabled.

**Exit criterion**: `pytest` passes on schema round-trips; `llm.py` can make
a cached call.

## Phase 1 — End-to-end with hand-written prompts (3 days)

- `generator.py`, `distinguisher.py`, `orchestrator.py` with one fixed prompt
  each.
- `user_proxy.py` as an oracle (gold-spec-aware, minimal-answer policy).
- Run on 5 seed tasks. Read every trace by hand.

**Exit criterion**: traces look sensible to a human reviewer. No reward
computation yet — we are sanity-checking that the loop produces meaningful
specs at all.

## Phase 2 — External benchmark integration (3–4 days)

Code execution and grading are intentionally out of scope for this project.
We wrap an off-the-shelf benchmark library that already does both.
Default: **EvalPlus** (MBPP+ / HumanEval+). Swap by changing the adapter.

- Add the library as a dependency.
- `src/idcs/benchmark/tasks.py` — thin adapter that loads benchmark tasks
  from the library and normalizes them to our `Task` schema.
- `src/idcs/coder.py` (frozen prompt) — Spec → Python source. This is the
  "model" the library scores.
- `src/idcs/benchmark/scoring.py` — wrap the library's grader call so the
  rest of the pipeline gets a single `score(task, code) -> float`.
- Establish two baselines:
  - (a) input → code directly (no spec)
  - (b) input → spec → code with frozen prompts

**Exit criterion**: (b) beats (a) on the library's scoring. If it doesn't,
fix the spec format before adding any optimization machinery.

## Phase 3 — Rewards + telemetry (2 days)

- `rewards.py` implementing R_G, R_D.
- `attribution.py` for counterfactual type-2 credit.
- `telemetry.py` emitting JSONL of every turn.
- `scripts/inspect.py` for browsing runs.

**Exit criterion**: reward values move sensibly on hand-edited prompt
variants (e.g. a deliberately bad G prompt scores lower).

## Phase 4 — Single-side optimization (3–4 days)

- `optimizer/population.py`, `optimizer/mutate.py`.
- Freeze D, optimize G via population search. Then freeze G, optimize D.

**Exit criterion**: each side independently moves the benchmark score on a
held-out task batch. This phase catches reward bugs before coevolution
amplifies them.

## Phase 5 — Coevolution (~1 week)

- `optimizer/coevolve.py`: alternating G/D epochs.
- Diversity guards (pairwise edit distance within each population).
- Anti-collapse monitors: spec-length distribution, type-2 rate per episode,
  embedding distance from input.

**Exit criterion**: stable training run over ~50 epochs without collapse,
with final populations beating the Phase 2 fixed-prompt baseline by a
meaningful margin on held-out tasks.

## Phase 6 — Real user evaluation (~1 week)

- Swap oracle `user_proxy` for a human-in-the-loop interface.
- Measure: spec quality, questions per task, dismissal rate, time-to-spec.
- Compare against the Phase 2 fixed-prompt baseline.

**Exit criterion**: real-user metrics validate the offline gains, or surface
a clear oracle-vs-human distribution gap to fix.

## Risks and mitigations

1. **Oracle leakage** — user_proxy answers too completely; D learns to
   extract gold specs by asking. Mitigate by minimal-answer policy with
   noise injection.
2. **Benchmark gaming** — G/D learn MBPP+ quirks. Mitigate by holding out a
   second benchmark for eval-only; rotate during training.
3. **Type-2 inflation despite penalty** — small benchmark wins per question
   tempt D to spam. Mitigate by capping questions per episode and
   penalizing on the cap, not the average.
4. **Cold-start emptiness** — D returns ∅ immediately because initial specs
   look fine to it. Mitigate by bootstrapping D on synthetic-gap-injected
   gold specs before any coevolution.

## Open questions to revisit

- Spec schema rigidity: too loose and the coder can't use it; too rigid and
  G can't express edge cases. Plan to iterate on this during Phase 1–2.
- Whether to share a single LLM call between G and D (with role headers) or
  keep them strictly separate. Default: separate, for clean caching.
- How to handle multi-turn user clarifications during a single episode
  (batch all type-2 issues per turn, or interleave?). Default: batch per
  turn.
