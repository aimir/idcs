# Overview

## Goal

A pipeline for **formal-spec-guided software development**: given a natural-language
task description, produce a structured specification that is faithful enough to
the user's intent to drive reliable code generation.

The pipeline is built around two cooperating-but-adversarial LLM roles, each
realized as a system prompt over an existing pre-trained model:

- **Generator (G)**: input → structured spec, and (spec, issues, answers) →
  revised spec.
- **Distinguisher (D)**: input × spec → list of issues, where each issue is
  classified by routing:
  - **Type 1 (generator-routed)**: a true gap or contradiction the generator
    should fix without bothering the user.
  - **Type 2 (user-routed)**: under-specification where additional input from
    the user would meaningfully improve the spec (e.g. crucial safety
    constraints).

## Co-optimization

G and D are optimized simultaneously. Because we are optimizing prompts (not
weights), the outer loop is a population-based prompt search rather than RL on
parameters.

### Reward shape (intuition; precise form in `design.md`)

- **Benchmark score on final spec** dominates both players' rewards. Without a
  strong anchor, adversarial training collapses.
- **G** is penalized per type-1 issue raised against it (so it learns to
  produce specs that survive scrutiny on the first pass).
- **D** is rewarded per type-1 issue that G actually fixes (credit only for
  *accepted* rejects, not raw nitpicks).
- **Type-2 issues** are rewarded via a **useful-clarification rate**: the
  benchmark-score delta attributable to each user question, divided by the
  number of questions asked. This naturally penalizes spam — a question that
  doesn't change the spec in a way that improves the score earns nothing.

## Why this shape

The two main failure modes of naive adversarial setups are:

1. **Generator collapse**: G learns to echo the input or produce trivial specs
   that D can't reject. Mitigated by the benchmark anchor and a
   spec-complexity floor.
2. **Distinguisher inflation**: D learns to nitpick endlessly. Mitigated by
   only rewarding *accepted* rejects (type 1) and *useful* clarifications
   (type 2).

The benchmark must be strong enough that both players' best path to high
reward goes through producing genuinely good specs, not through exploiting
the adversarial shaping terms.

## Why prompts, not weights

- Cheap to iterate; no GPUs to manage.
- Tractable search space (DSPy / GEPA / evolutionary prompt mutation).
- Failures are inspectable — a prompt is text you can read.
- Risk: prompt search has a lower ceiling than fine-tuning. Acceptable for
  a first iteration; revisit if results plateau.

## Status

Implementation matches the design through `plan.md` Phase 5. What's live today:

- **Pipeline**: generator, distinguisher, oracle user-proxy, coder,
  orchestrator. Per-trace structured telemetry with prompt hashes and
  config snapshot.
- **Benchmarks**: EvalPlus / MBPP+ adapter with `plus_input` scoring;
  hard-train / hard-dev / hard-test splits for held-out generalization;
  a curated `hardened` POC corpus separating underspecification rescue
  from raw difficulty.
- **LLM backends**: OpenRouter (default), OpenAI, and a local Codex CLI
  backend with budgeted retries, exponential backoff, JSON-repair retry
  for malformed structured output, and budget-safe error semantics.
- **Optimizer**: population-based G/D coevolution with anchor-protected
  base prompt, task-Pareto elite selection, diversity guard, plain-text
  mutator fallback, anti-regression penalty against the direct baseline,
  and held-out validation split.
- **Findings**: the hand-written spec-guided pipeline raises hidden-test
  pass rate 73.3% → 96.2% on a held-out hard-test split; the diagnostic
  hand-rules ceiling reaches 100% on the original 5-task hard slice;
  coevolved prompts transfer to training tasks but not yet to the
  held-out split. See `apart-hackathon-submission.md` for the full
  evidence ledger.

What's not yet built: real-user evaluation (`plan.md` Phase 6).
