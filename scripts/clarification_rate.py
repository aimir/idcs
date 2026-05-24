"""Measure useful-clarification-rate on the seed corpus.

For each seed task, run both baselines:

- (a) prompt → code → score (no spec layer)
- (b) prompt → spec via G/D with OracleUserProxy (reads gold_spec) → code → score

Per task we compute ``delta = score_b - score_a`` and ``clarifications_b`` =
number of type-2 issues the oracle answered. Aggregate:

    useful_clarification_rate = sum(delta) / sum(clarifications)

A positive rate means each oracle answer, on average, improved benchmark
score by this much. This is a noisier proxy than the counterfactual
attribution design.md describes (Phase 3 work), but it gives Phase 2 a
quantitative signal that the spec/clarification loop is doing its job.

Requires ``OPENROUTER_API_KEY``. Optional ``IDCS_MODEL`` to override
the default model.

Usage:
    python scripts/clarification_rate.py                  # all 8 seed tasks
    python scripts/clarification_rate.py --tasks 04 05    # subset by id substring
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idcs.benchmark.scoring import score  # noqa: E402
from idcs.coder import Coder  # noqa: E402
from idcs.distinguisher import Distinguisher  # noqa: E402
from idcs.generator import Generator  # noqa: E402
from idcs.llm import LLM  # noqa: E402
from idcs.orchestrator import run_episode  # noqa: E402
from idcs.seed_corpus import load_seed_corpus  # noqa: E402
from idcs.user_proxy import OracleUserProxy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="task-id substrings to filter (e.g. 01 04)",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    corpus = load_seed_corpus()
    if args.tasks:
        corpus = [it for it in corpus if any(s in it.task.id for s in args.tasks)]
    if not corpus:
        print("No seed tasks matched.", file=sys.stderr)
        return 1

    llm = LLM()
    print(f"\n{'=' * 60}")
    print(f"Model:      {llm.model}")
    print(f"Seed tasks: {len(corpus)}")
    print(f"Max turns:  {args.max_turns}")
    print(f"{'=' * 60}\n")

    coder = Coder(llm)
    generator = Generator(llm)
    distinguisher = Distinguisher(llm)

    results: list[dict[str, float | int | str]] = []
    t_start = time.time()
    for i, item in enumerate(corpus, 1):
        task = item.task
        oracle = OracleUserProxy(
            llm,
            gold_spec_text=item.gold_spec.model_dump_json(indent=2),
        )
        print(f"[{i}/{len(corpus)}] {task.id}")
        t0 = time.time()

        code_a = coder.from_prompt(task.prompt)
        score_a = score(task, code_a)

        trace = run_episode(task, generator, distinguisher, oracle, max_turns=args.max_turns)
        clar_count = sum(len(t.user_answers) for t in trace.turns)
        if trace.final_spec is None:
            score_b = 0.0
        else:
            code_b = coder.from_spec(trace.final_spec, task.prompt)
            score_b = score(task, code_b)

        delta = score_b - score_a
        elapsed = time.time() - t0
        tag = "WIN" if delta > 0 else ("LOSE" if delta < 0 else "TIE")
        results.append(
            {"task_id": task.id, "a": score_a, "b": score_b, "delta": delta, "clar": clar_count}
        )
        print(
            f"  => (a)={score_a:.2f}  (b)={score_b:.2f}  "
            f"Δ={delta:+.2f}  clar={clar_count}  {tag}  [{elapsed:.1f}s]\n"
        )

    total_time = time.time() - t_start
    avg_a = sum(float(r["a"]) for r in results) / len(results)
    avg_b = sum(float(r["b"]) for r in results) / len(results)
    total_delta = sum(float(r["delta"]) for r in results)
    total_clar = sum(int(r["clar"]) for r in results)

    print(f"{'=' * 60}")
    print(f"Aggregate:  (a)={avg_a:.3f}  (b)={avg_b:.3f}  Δavg={avg_b - avg_a:+.3f}")
    print(f"Total clarifications: {total_clar}")
    if total_clar > 0:
        rate = total_delta / total_clar
        print(
            f"useful_clarification_rate = Σ(Δ)/Σ(clarifications) "
            f"= {total_delta:+.3f} / {total_clar} = {rate:+.4f}"
        )
    else:
        print("useful_clarification_rate = N/A (no user clarifications were answered)")
    print(f"Time: {total_time:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
