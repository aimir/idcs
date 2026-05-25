# Adversarial Iteration for Underspecified Program Synthesis

[![CI](https://github.com/aimir/idcs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aimir/idcs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Apart Research](https://img.shields.io/badge/Apart%20Research-Hackathon%20Submission-7e3ff2.svg)](https://apartresearch.com/project/adversarial-iteration-for-underspecified-program-synthesis-i82n)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)](https://github.com/aimir/idcs/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict)

LLM-generated code routinely passes the obvious examples and silently fails the hidden ones. **IDCS** (Iterative Distinguishing of Code and Specs) treats this as a *specification* problem rather than a model problem: a **generator** drafts a structured spec from the prompt, a **distinguisher** critiques it for gaps and ambiguities, a **user-proxy** answers the questions that need user intent, and a **coder** implements the final spec. On a held-out hard MBPP+ split, this spec-guided pipeline raises hidden-test pass rate from **73.3% → 96.2%**. We also show that the generator and distinguisher prompts themselves can be discovered through adversarial *coevolution*.

📄 **[Apart Research submission](https://apartresearch.com/project/adversarial-iteration-for-underspecified-program-synthesis-i82n)** • 📚 **[Paper](docs/apart-hackathon-submission.md)** • 🏗 **[Design notes](docs/design.md)** • 📋 **[Implementation plan](docs/plan.md)**

## Setup

```bash
git clone https://github.com/aimir/idcs.git
cd idcs
pip install -e ".[dev]"   # or: pip install -r requirements.txt
```

Python 3.11+ required.

### Environment variables

Put these in a `.env` file at the repo root (auto-loaded by [`python-dotenv`](https://pypi.org/project/python-dotenv/)):

```bash
# Default backend: OpenRouter (any OpenAI-compatible provider works)
OPENROUTER_API_KEY=sk-or-v1-...
IDCS_MODEL=anthropic/claude-sonnet-4.5     # optional, defaults to claude-sonnet-4.5

# Or use a local Codex CLI install (no API key needed)
IDCS_BACKEND=codex
IDCS_CODEX_MODEL=gpt-5.4-mini              # optional
IDCS_CODEX_TIMEOUT_S=300                   # optional, per-call timeout
IDCS_CODEX_SERVICE_TIER=fast               # optional
IDCS_CODEX_REASONING_EFFORT=none           # optional
```

Switch backends with `IDCS_BACKEND`: `openrouter` (default), `openai`, `openai-compatible`, or `codex`. Use a different mutator model for prompt search with `IDCS_MUTATOR_MODEL`.

## Examples

### Smoke: direct vs spec-guided on one MBPP+ task

```bash
python scripts/baseline.py --tasks Mbpp/11 --max-turns 1
```

Output is a side-by-side pass-rate comparison on the hidden `plus_input` cases.

### Batch evaluation on the hard slice

```bash
python scripts/batch_baseline.py --dataset hard --workers 4 --max-turns 2
```

Writes incremental `results.jsonl` and live-updated `summary.json` to `experiments/runs/`. Hard slices available: `hard`, `hard-train`, `hard-dev`, `hard-test`, `hard-extended`, plus the curated `hardened` POC corpus.

### Coevolution training

```bash
python scripts/train.py \
  --benchmark hard \
  --epochs 5 --pop-size 8 --elite-size 3 \
  --val-fraction 0.2 \
  --max-llm-calls 500
```

Evolves generator and distinguisher prompts via population-based search with anchor-protected elites, task-Pareto elite selection, and an anti-regression reward penalty. Telemetry (per-turn traces, prompt hashes, config snapshot) lands under `experiments/runs/<run_id>/`.

### Reproducing the ceiling result

The diagnostic hand-rules prompts that reach 100% on the hard slice:

```bash
IDCS_BACKEND=codex IDCS_CODEX_MODEL=gpt-5.4-mini \
python scripts/batch_baseline.py \
  --dataset hard --workers 5 --retries 0 --max-turns 2 \
  --generator-prompt-file prompts/hard_mbpp_rules_generator_v0.md \
  --distinguisher-prompt-file prompts/hard_mbpp_rules_distinguisher_v0.md
```

## Development

```bash
pytest -q              # 112 tests
ruff check src tests scripts
mypy src/idcs scripts
```

## Authors

Uri Ariel Chen · Nitzan Pomerantz · Amir Sarid · Amit Saroussi — all independent researchers, with [Apart Research](https://apartresearch.com).
