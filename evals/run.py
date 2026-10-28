"""Run the evaluation scenarios and write a report.

    python -m evals.run --llm oracle                      # harness check, free, instant
    python -m evals.run --llm naive                       # must FAIL: shows the checks catch bad behaviour
    python -m evals.run --llm anthropic --model claude-opus-5-5 --repeat 3   # real model; needs credentials, costs money
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from agent.core import Agent
from agent.crm import InMemoryCRM

from .policies import scripted
from .scenarios import SCENARIOS, Scenario


def run_scenario(scenario: Scenario, llm) -> dict:
    crm = InMemoryCRM()  # fresh, identical starting state for every run
    started = time.time()
    result = Agent(llm, crm).run(scenario.prompt)
    checks = scenario.check(result, crm)
    return {
        "id": scenario.id, "category": scenario.category, "prompt": scenario.prompt,
        "passed": all(checks.values()), "checks": checks,
        "tool_calls": [{"name": c.name, "input": c.input, "error": c.is_error, "blocked_by": c.blocked_by}
                       for c in result.tool_calls],
        "guardrail_blocks": sum(1 for c in result.tool_calls if c.blocked_by),
        "writes": result.writes, "steps": result.steps, "stop_reason": result.stop_reason,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "seconds": round(time.time() - started, 2), "final_text": result.final_text,
    }


def summarise(rows: list[dict]) -> dict:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["passed"])
    n = len(rows)
    return {
        "runs": n, "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3) if n else 0.0,
        "by_category": {c: {"passed": sum(v), "runs": len(v)} for c, v in sorted(by_cat.items())},
        "guardrail_blocks": sum(r["guardrail_blocks"] for r in rows),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2) if n else 0.0,
        "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def markdown(llm_name: str, summary: dict, rows: list[dict]) -> str:
    lines = [f"# Eval report: {llm_name}", "",
             f"**{summary['passed']}/{summary['runs']} passed ({summary['pass_rate']:.0%})**, "
             f"{summary['guardrail_blocks']} guardrail blocks, {summary['avg_steps']} steps on average, "
             f"{summary['input_tokens']:,} input / {summary['output_tokens']:,} output tokens.", "",
             "| Category | Passed |", "|---|---|"]
    lines += [f"| {c} | {v['passed']}/{v['runs']} |" for c, v in summary["by_category"].items()]
    lines += ["", "| Scenario | Result | Failed checks | Steps | Blocks |", "|---|---|---|---|---|"]
    for r in rows:
        failed = ", ".join(k for k, ok in r["checks"].items() if not ok) or "-"
        lines.append(f"| {r['id']} | {'pass' if r['passed'] else 'FAIL'} | {failed} | {r['steps']} | {r['guardrail_blocks']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run CRM support agent evals")
    parser.add_argument("--llm", choices=["oracle", "naive", "anthropic"], default="oracle")
    parser.add_argument("--model", default="claude-opus-5-5", help="model id when --llm anthropic")
    parser.add_argument("--repeat", type=int, default=1, help="runs per scenario (models are not deterministic)")
    parser.add_argument("--scenario", action="append", help="only run this scenario id (repeatable)")
    parser.add_argument("--out", default="eval_results", help="directory for the JSON and Markdown reports")
    args = parser.parse_args(argv)

    chosen = [s for s in SCENARIOS if not args.scenario or s.id in args.scenario]
    if not chosen:
        print("no scenarios selected", file=sys.stderr)
        return 2
    if args.llm == "anthropic":
        from agent.llm import AnthropicLLM
        shared = AnthropicLLM(model=args.model)
        make_llm = lambda scenario: shared  # noqa: E731
        name = args.model
    else:
        make_llm = lambda scenario: scripted(args.llm, scenario.id)  # noqa: E731
        name = f"{args.llm} (scripted, not a model)"

    rows = []
    for scenario in chosen:
        for _ in range(args.repeat):
            row = run_scenario(scenario, make_llm(scenario))
            rows.append(row)
            print(f"{row['id']:>3} {'pass' if row['passed'] else 'FAIL'}  steps={row['steps']} blocks={row['guardrail_blocks']}"
                  + ("" if row["passed"] else "  failed: " + ", ".join(k for k, ok in row["checks"].items() if not ok)))

    summary = summarise(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.llm}" + (f"_{args.model}" if args.llm == "anthropic" else "")
    (out / f"{stem}.json").write_text(json.dumps({"llm": name, "summary": summary, "runs": rows}, indent=2), encoding="utf8")
    (out / f"{stem}.md").write_text(markdown(name, summary, rows), encoding="utf8")
    print(f"\n{summary['passed']}/{summary['runs']} passed ({summary['pass_rate']:.0%}); report in {out}/{stem}.md")
    return 0 if summary["passed"] == summary["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
