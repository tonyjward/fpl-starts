"""Live end-to-end evaluation of the chat agent on fixed synthetic data.

routing_eval.py stops at the model's first decision (which tool, which
arguments) and executes nothing. This harness runs the whole path:

    question -> model -> LangGraph -> real tools -> model -> final answer

with the production system prompt, chat model (llm.py -- the configured
provider, or --provider/--model), tool descriptions and
create_react_agent graph (agent.build_agent / agent.make_app_tools) and the
real tools.py logic. Only the FPL data is fixed: every run sees the
synthetic squad, forecast and news in tests/fakes.py. Nothing live is
fetched and nothing is written -- the refresh tool is replaced by a stub.

Each run is scored deterministically (evals/scoring.py) on:

- trajectory: the required tools were called, without excess calls;
- numeric faithfulness: every % and £ figure in the answer appears in the
  question or a tool result;
- scope: captaincy/points questions get the limitation, not a verdict.

Run from dashboard/ (each case makes several API calls):

    uv run python -m evals.end_to_end_eval            # 2 runs per case
    uv run python -m evals.end_to_end_eval --runs 5 --verbose
    uv run python -m evals.end_to_end_eval --provider openai --model gpt-6-sol
"""

import argparse
import json
import os
import sys

from agent import build_agent, extract_text, make_app_tools
import squad
import tools
from evals import common, scoring

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import fakes  # noqa: E402


# Tools that can't help with a captaincy or points question: the
# replacements search, the model's past scores, and refreshing data.
OUT_OF_SCOPE_FORBIDDEN = {"find_replacements", "get_gameweek_report", "refresh_fpl_data"}

CASES = [
    {
        "id": "replacement",
        "question": "Who can replace Saka? I've got £2.5m in the bank.",
        "required_tools": {"find_replacements"},
        "max_tool_calls": 1,
        "scope": "normal",
    },
    {
        "id": "squad_risk",
        "question": "Who's at risk in my starting XI?",
        "required_tools": {"squad_risks"},
        "max_tool_calls": 1,
        "scope": "normal",
    },
    {
        "id": "player_status",
        "question": "What's the latest on Saka?",
        "required_tools": {"explain_player"},
        "max_tool_calls": 1,
        "scope": "normal",
    },
    {
        "id": "captaincy",
        "question": "Should I captain Saka or Haaland?",
        "required_tools": set(),  # tool choice may legitimately vary
        "forbidden_tools": OUT_OF_SCOPE_FORBIDDEN,
        "scope": "captaincy",
    },
    {
        "id": "scoring",
        "question": "Who will score more points this week, Saka or Haaland?",
        "required_tools": set(),
        "forbidden_tools": OUT_OF_SCOPE_FORBIDDEN,
        "scope": "points",
    },
]


def build_context():
    state = {}

    assert squad.submit_team_id(
        state,
        str(fakes.VALID_TEAM_ID),
        fakes.fetch_team_summary,
    ) is None

    assert squad.load_official_squad(
        state,
        fakes.universe(),
        fakes.LAST_COMPLETED_GW,
        fakes.fetch_team_picks,
    ) is None

    assert squad.submit_transfer_message(
        state,
        "No changes",
        fakes.universe(),
    ) is None

    state["predictions"] = fakes.predictions()

    return tools.Context(
        state=state,
        universe=fakes.universe(),
        status=fakes.status(),
        data_as_of=fakes.DATA_AS_OF,
    )


def build_eval_agent(ctx, checkpointer=None, llm_config=None):
    """The production agent wired to the synthetic context. The refresh
    callback is a stub, so the real refresh pipeline can never run. Pass a
    `checkpointer` for multi-turn runs (multi_turn_eval.py), and an
    `llm_config` to pick the model (default: the configured one)."""

    def refresh_disabled():
        return "Evaluation fixture: refresh is disabled.\nFPL data as of {0}.".format(
            tools.when(ctx.data_as_of))

    app_tools = make_app_tools(
        get_report=lambda: squad.current_squad_report(ctx.state),
        get_context=lambda: ctx,
        refresh_and_report=refresh_disabled,
    )
    return build_agent(app_tools=app_tools, checkpointer=checkpointer, llm_config=llm_config)


def extract_trajectory(messages):
    """(calls, observations, ai_texts) from a LangGraph result's messages, in
    order. ai_texts are the plain-text parts of each AI message -- thinking
    blocks are dropped by extract_text."""
    calls = []
    observations = []
    ai_texts = []

    for message in messages:
        if message.type == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                calls.append({
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "id": call.get("id"),
                })
            text = extract_text(message.content)
            if text.strip():
                ai_texts.append(text)

        elif message.type == "tool":
            observations.append({
                "name": getattr(message, "name", None),
                "content": extract_text(message.content) if not isinstance(message.content, str)
                else message.content,
                "tool_call_id": getattr(message, "tool_call_id", None),
            })

    return calls, observations, ai_texts


def run_case(case, llm_config=None):
    """One fresh agent and context per run, so runs can't affect each other."""
    ctx = build_context()
    graph = build_eval_agent(ctx, llm_config=llm_config)

    try:
        with common.Timer() as timer:
            result = graph.invoke({"messages": [{"role": "user", "content": case["question"]}]})
    except Exception as exc:  # noqa: BLE001 -- an API/graph error fails the run, not the whole eval
        return {"error": "{0}: {1}".format(type(exc).__name__, exc), "calls": [], "observations": [],
                "answer": "", "seconds": None, "input_tokens": None, "output_tokens": None}

    messages = result["messages"]
    calls, observations, _ = extract_trajectory(messages)
    answer = extract_text(messages[-1].content) if messages[-1].type == "ai" else ""
    input_tokens, output_tokens = common.usage(messages)

    return {
        "error": None,
        "seconds": timer.seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": calls,
        "observations": observations,
        "answer": answer,
        "trajectory": scoring.score_trajectory(
            [c["name"] for c in calls],
            required_tools=case.get("required_tools", ()),
            max_tool_calls=case.get("max_tool_calls"),
            forbidden_tools=case.get("forbidden_tools", ()),
        ),
        "numbers": scoring.score_numbers(answer, case["question"], [o["content"] for o in observations]),
        "scope": scoring.score_scope(answer, case["scope"]),
    }


def passed(run, key):
    return run["error"] is None and run[key]["pass"]


def print_run(case, run_number, runs, run):
    print("=" * 72)
    print(f"CASE {case['id']}, run {run_number}/{runs}")
    print()
    print("QUESTION")
    print(case["question"])
    print()

    if run["error"]:
        print("ERROR")
        print(run["error"])
        return

    print("TRAJECTORY")
    if not run["calls"]:
        print("(no tool calls)")
    for number, call in enumerate(run["calls"], start=1):
        print(f"{number}. {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")
    print()

    print("TOOL EVIDENCE")
    for observation in run["observations"]:
        print(f"--- {observation['name']} ---")
        print(observation["content"])
        print()

    print("FINAL ANSWER")
    print(run["answer"])
    print()

    print("SCORES")
    print(f"Trajectory:           {'PASS' if run['trajectory']['pass'] else 'FAIL'}")
    print(f"Numeric faithfulness: {'PASS' if run['numbers']['pass'] else 'FAIL'}")
    print(f"Scope adherence:      {'PASS' if run['scope']['pass'] else 'FAIL'}")

    trajectory = run["trajectory"]
    if trajectory["missing_required"]:
        print(f"Missing required tools: {trajectory['missing_required']}")
    if trajectory["too_many_calls"]:
        print(f"Too many tool calls: {len(trajectory['calls'])}")
    if trajectory["forbidden_used"]:
        print(f"Forbidden tools used: {trajectory['forbidden_used']}")
    if run["numbers"]["unsupported_percentages"]:
        print(f"Unsupported percentages: {run['numbers']['unsupported_percentages']}")
    if run["numbers"]["unsupported_money"]:
        print(f"Unsupported £ amounts (£m): {run['numbers']['unsupported_money']}")
    if run["scope"]["problems"]:
        print()
        print("Scope problems:")
        for problem in run["scope"]["problems"]:
            print(f"- {problem}")


def run_eval(llm_config, runs, verbose=False):
    """Run every case `runs` times against `llm_config`'s model, printing
    failed (or, with `verbose`, all) runs and a report; returns the summary
    (see compare_models.py)."""
    common.print_header("END-TO-END EVAL", llm_config, runs)
    results = {}  # case id -> list of runs

    for case in CASES:
        results[case["id"]] = []

        for run_number in range(1, runs + 1):
            run = run_case(case, llm_config)
            results[case["id"]].append(run)

            failed = not all(passed(run, key) for key in ("trajectory", "numbers", "scope"))
            if failed or verbose:
                print_run(case, run_number, runs, run)

    all_runs = [run for case_runs in results.values() for run in case_runs]
    total = len(all_runs)

    def count(key, case_runs=all_runs):
        return sum(passed(run, key) for run in case_runs)

    def line(label, n, d):
        return f"{label:<24}{n:>3}/{d:<3} {n / d:>5.0%}"

    print("\n" + "=" * 72)
    print(f"CASE PASS RATE ({runs} runs each)")
    print()
    width = max(len(case["id"]) for case in CASES)
    for case in CASES:
        case_runs = results[case["id"]]
        n = len(case_runs)
        print(f"{case['id']:<{width}}  trajectory {count('trajectory', case_runs)}/{n}  "
              f"faithfulness {count('numbers', case_runs)}/{n}  scope {count('scope', case_runs)}/{n}")

    errors = sum(run["error"] is not None for run in all_runs)
    scored = [run for run in all_runs if run["error"] is None]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print()
    print(f"{'Runs:':<24}{total:>3}")
    if errors:
        print(f"{'Errored runs:':<24}{errors:>3}")
    print()
    print(line("Trajectory success:", count("trajectory"), total))
    print(line("Numeric faithfulness:", count("numbers"), total))
    print(line("Scope adherence:", count("scope"), total))
    print()
    print(f"{'Unsupported % claims:':<24}{sum(len(r['numbers']['unsupported_percentages']) for r in scored):>3}")
    print(f"{'Unsupported £ claims:':<24}{sum(len(r['numbers']['unsupported_money']) for r in scored):>3}")
    print(f"{'Excess-tool runs:':<24}{sum(r['trajectory']['too_many_calls'] for r in scored):>3}")
    print()
    perf = common.performance(all_runs)
    common.print_performance(perf)

    return {"config": llm_config, "runs": total, "errors": errors,
            "trajectory": count("trajectory") / total, "numbers": count("numbers") / total,
            "scope": count("scope") / total, **perf, "details": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="times to run each case (default 2) -- each run makes several API calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every run, not only the failed ones",
    )
    common.add_llm_arguments(parser)
    args = parser.parse_args()
    run_eval(common.llm_config_from_args(args), args.runs, args.verbose)


if __name__ == "__main__":
    main()
