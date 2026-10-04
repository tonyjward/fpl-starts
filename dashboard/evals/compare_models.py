"""Run the same evals against several models and compare them side by side.

No eval logic of its own: for each model it runs routing_eval,
end_to_end_eval and multi_turn_eval exactly as their own commands would
(failures are printed as they happen), then tabulates their summaries.
Same prompt, tools, golden cases and scorers -- only the model differs.

Run from dashboard/ (this makes many API calls -- check --runs):

    uv run python -m evals.compare_models \\
        --model anthropic:claude-opus-5 --model openai:gpt-6-sol --runs 5

Latency is wall-clock; tokens are the providers' own reported usage. No
cost is estimated: prices aren't in the API responses, so a cost figure
would be a guess.
"""

import argparse

from evals import end_to_end_eval, multi_turn_eval, routing_eval
from llm import load_llm_config


def _parse(spec):
    provider, _, model = spec.partition(":")
    if not model:
        raise argparse.ArgumentTypeError("use provider:model, e.g. openai:gpt-6-sol")
    return load_llm_config(provider, model)


def _pct(value):
    return "n/a" if value is None else f"{value:.0%}"


def _num(value, spec):
    return "n/a" if value is None else format(value, spec)


ROWS = [
    ("Routing: tool", "routing", "tool", _pct),
    ("Routing: arguments", "routing", "args", _pct),
    ("Routing: median latency (s)", "routing", "median_seconds", lambda v: _num(v, ".1f")),
    ("E2E trajectory", "e2e", "trajectory", _pct),
    ("E2E numeric faithfulness", "e2e", "numbers", _pct),
    ("E2E scope adherence", "e2e", "scope", _pct),
    ("E2E median latency (s)", "e2e", "median_seconds", lambda v: _num(v, ".1f")),
    ("E2E mean input tokens", "e2e", "mean_input_tokens", lambda v: _num(v, ",.0f")),
    ("E2E mean output tokens", "e2e", "mean_output_tokens", lambda v: _num(v, ",.0f")),
    ("Multi-turn continuity", "multi", "continuity", _pct),
    ("Multi-turn turn-2 trajectory", "multi", "trajectory", _pct),
    ("Multi-turn median latency (s)", "multi", "median_seconds", lambda v: _num(v, ".1f")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", dest="models", type=_parse, action="append", required=True,
                        help="provider:model to evaluate; repeat for each model")
    parser.add_argument("--runs", type=int, default=2, help="runs per case for every eval (default 2)")
    args = parser.parse_args()

    summaries = []
    for config in args.models:
        summaries.append({
            "routing": routing_eval.run_eval(config, args.runs),
            "e2e": end_to_end_eval.run_eval(config, args.runs),
            "multi": multi_turn_eval.run_eval(config, args.runs),
        })
        print()

    labels = [config.label for config in args.models]
    width = 5  # wide enough for "n/a" plus spacing
    for label in labels:
        width = max(width, len(label) + 2)

    def row(title, cells):
        line = f"{title:<32}"
        for cell in cells:
            line += f"{cell:>{width}}"
        print(line)

    print("=" * 72)
    print(f"MODEL COMPARISON ({args.runs} runs per case)")
    print()
    row("", labels)
    for title, eval_name, key, fmt in ROWS:
        cells = []
        for summary in summaries:
            cells.append(fmt(summary[eval_name].get(key)))
        row(title, cells)

    errors = []
    for summary in summaries:
        errors.append(summary["e2e"]["errors"] + summary["multi"]["errors"])
    if sum(errors):
        row("Errored runs (E2E + multi)", errors)


if __name__ == "__main__":
    main()
