"""Live evaluation of the agent's first routing decision.

Uses the real production system prompt, tool schemas and chat model (the
same llm.build_chat_model as the app, so any configured provider), but
does NOT execute any tools. This evaluates:

- tool selection
- argument extraction

Run from dashboard/:

    uv run python -m evals.routing_eval            # 5 runs per case
    uv run python -m evals.routing_eval --runs 10
    uv run python -m evals.routing_eval --provider openai --model gpt-6-sol
"""

import argparse
from math import isclose

from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
    SYSTEM_PROMPT,
    extract_text,
    get_gameweek_report,
    make_app_tools,
)
from evals import common
from llm import build_chat_model


CASES = [
    {
        "question": "Who's at risk in my squad?",
        "expected_tool": "squad_risks",
        "expected_args": {},
    },
    {
        "question": "Anyone dodgy in my starting XI?",
        "expected_tool": "squad_risks",
        "expected_args": {},
    },
    {
        "question": "Why is Palmer only 80% likely to start?",
        "expected_tool": "explain_player",
        "expected_args": {"name": "Palmer"},
    },
    {
        "question": "Is Palmer fit?",
        "expected_tool": "explain_player",
        "expected_args": {"name": "Palmer"},
    },
    {
        "question": "Any injury news in my team?",
        "expected_tool": "player_news",
        "expected_args": {},
    },
    {
        "question": "Who can replace Saka? I've got £2.5m in the bank.",
        "expected_tool": "find_replacements",
        "expected_args": {
            "replacing": "Saka",
            "bank": 2.5,
        },
    },
    {
        "question": "Find me a defender under £5m who's at least 80% likely to start.",
        "expected_tool": "find_replacements",
        "expected_args": {
            "position": "defender",
            "max_price": 5.0,
            "min_chance": 0.8,
        },
    },
    {
        "question": "How did the model do in gameweek 5?",
        "expected_tool": "get_gameweek_report",
        "expected_args": {"target_round": 5},
    },
    {
        "question": "Is your FPL data up to date?",
        "expected_tool": "refresh_fpl_data",
        "expected_args": {},
    },
]


def build_router(llm_config):
    """Build the same model + tool interface used by the real app.

    The callback bodies are deliberately harmless: this script never
    executes the tools. We only ask the model which tool it would call.
    """

    llm = build_chat_model(llm_config)

    app_tools = make_app_tools(
        get_report=lambda: "stub",
        get_context=lambda: None,
        refresh_and_report=lambda: "stub",
    )

    tools = [get_gameweek_report] + app_tools

    return llm.bind_tools(tools)


# Player-name arguments: the model may expand "Palmer" to "Cole Palmer",
# which the tools resolve just the same.
NAME_ARGS = {"name", "replacing"}


def value_matches(key, actual, expected):
    if isinstance(expected, float):
        try:
            return isclose(float(actual), expected, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False

    if key in NAME_ARGS and isinstance(actual, str):
        # Every word of the expected name appears in the actual one.
        return set(expected.lower().split()) <= set(actual.lower().split())

    return actual == expected


def expected_args_match(actual, expected):
    """Expected args are a subset.

    The model is allowed to explicitly send optional defaults that we did
    not specify in the golden case.
    """

    return all(
        key in actual and value_matches(key, actual[key], value)
        for key, value in expected.items()
    )


def run_case(router, case):
    """(tool_ok, args_ok, actual_tool, actual_args, calls, performance) for
    one call; performance is {seconds, input_tokens, output_tokens}."""

    with common.Timer() as timer:
        response = router.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=case["question"]),
            ]
        )
    input_tokens, output_tokens = common.usage([response])
    performance = {"seconds": timer.seconds, "input_tokens": input_tokens, "output_tokens": output_tokens,
                   "text": extract_text(response.content)}  # what it said instead, when it called no tool

    calls = response.tool_calls

    if calls:
        first = calls[0]
        actual_tool = first["name"]
        actual_args = first["args"]
    else:
        actual_tool = None
        actual_args = {}

    tool_ok = actual_tool == case["expected_tool"]
    args_ok = (
        tool_ok
        and expected_args_match(
            actual_args,
            case["expected_args"],
        )
    )

    return tool_ok, args_ok, actual_tool, actual_args, calls, performance


def run_eval(llm_config, runs):
    """Run every case `runs` times against `llm_config`'s model, printing
    failures and a report; returns the summary (see compare_models.py)."""
    common.print_header("ROUTING EVAL", llm_config, runs)
    router = build_router(llm_config)

    results = []  # per case: (tool passes, args passes)
    records = []  # per call: latency and usage

    for number, case in enumerate(CASES, start=1):
        tool_passes = 0
        args_passes = 0

        for run in range(1, runs + 1):
            tool_ok, args_ok, actual_tool, actual_args, calls, performance = run_case(router, case)
            performance.update({
                "case": number,
                "question": case["question"],
                "expected_tool": case["expected_tool"],
                "actual_tool": actual_tool,
                "actual_args": actual_args,
                "tool_pass": tool_ok,
                "args_pass": args_ok,
                "tool_calls": len(calls),
            })
            records.append(performance)

            tool_passes += int(tool_ok)
            args_passes += int(args_ok)

            # Only failures (and extra parallel calls) are worth reading.
            if tool_ok and args_ok and len(calls) <= 1:
                continue

            print("=" * 72)
            print(f"CASE {number}, run {run}/{runs}")
            print(f"Question:       {case['question']}")
            print(f"Expected tool:  {case['expected_tool']}")
            print(f"Actual tool:    {actual_tool}")
            print(f"Expected args:  {case['expected_args']}")
            print(f"Actual args:    {actual_args}")
            print(f"Tool routing:   {'PASS' if tool_ok else 'FAIL'}")
            print(f"Arguments:      {'PASS' if args_ok else 'FAIL'}")

            if len(calls) > 1:
                print(f"Extra calls:    {calls[1:]}")

        results.append((tool_passes, args_passes))

    print("\n" + "=" * 72)
    print(f"PASS RATE PER CASE ({runs} runs each)")
    print(f"{'#':>2}  {'tool':>5}  {'args':>5}  question")

    for number, (case, (tool_passes, args_passes)) in enumerate(
        zip(CASES, results), start=1
    ):
        flag = "" if args_passes == runs else "  <-- not 100%"
        print(
            f"{number:>2}  {tool_passes:>2}/{runs:<2}  {args_passes:>2}/{runs:<2}  "
            f"{case['question']}{flag}"
        )

    total = len(CASES) * runs
    tool_correct = sum(t for t, _ in results)
    args_correct = sum(a for _, a in results)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"Tool routing:     {tool_correct}/{total} ({tool_correct / total:.1%})")
    print(f"Argument routing: {args_correct}/{total} ({args_correct / total:.1%})")
    perf = common.performance(records)
    common.print_performance(perf, unit="call")

    return {"config": llm_config, "calls": total, "tool": tool_correct / total, "args": args_correct / total,
            **perf, "details": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="times to ask each question (default 5) -- the model's answers vary",
    )
    common.add_llm_arguments(parser)
    args = parser.parse_args()
    run_eval(common.llm_config_from_args(args), args.runs)


if __name__ == "__main__":
    main()
