"""Live multi-turn evaluation: does the agent carry a conversation?

end_to_end_eval.py asks one question per run. Here each case is a short
conversation on one agent, one synthetic context and one thread ID, with
an InMemorySaver checkpointer exactly as the app uses -- and the second
turn only makes sense given the first ("him", "yes"). Each turn sends only
its new message; the earlier ones come from the checkpointer.

Scored separately for the second turn:

- continuity: it acted on the first turn's context (the right player, the
  offered action) rather than asking what the user meant;
- trajectory, numeric faithfulness and scope, as in end_to_end_eval.py.

Run from dashboard/ (each case makes several API calls):

    uv run python -m evals.multi_turn_eval            # 2 runs per case
    uv run python -m evals.multi_turn_eval --runs 5 --verbose
    uv run python -m evals.multi_turn_eval --provider openai --model gpt-6-sol
"""

import argparse
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import extract_text, thread_config
from evals import common, scoring
from evals.end_to_end_eval import build_context, build_eval_agent, extract_trajectory


def _replaces_saka(calls):
    """A find_replacements call aimed at Saka: by name, or by his position
    (midfielder) -- either acts on what was just discussed."""
    return any(
        c["name"] == "find_replacements"
        and (scoring.names_match("Saka", c["args"].get("replacing"))
             or str(c["args"].get("position", "")).lower().startswith("mid"))
        for c in calls
    )


CASES = [
    {
        # "him" can't be resolved without the first turn.
        "id": "pronoun_follow_up",
        "turns": ["Tell me about Saka.", "Could you find replacements for him?"],
        "continuity": _replaces_saka,
        "required_tools": {"find_replacements"},
        "max_tool_calls": 1,
    },
    {
        # The production failure: the assistant offered replacements and the
        # user said "yes". The first exchange is seeded into the checkpointer
        # (as if turn 1 had happened), so the case doesn't depend on Claude
        # choosing to make the offer; only "yes" runs live.
        "id": "yes_after_offer",
        "seed": [
            ("human", "Who's at risk in my starting XI?"),
            ("ai", "Three players in your starting XI are below the 75% line:\n\n"
                   "- Saka (MID) -- 40% chance of starting, and FPL has him doubtful with a knock.\n"
                   "- Gvardiol (DEF) -- 55%.\n"
                   "- João Pedro (FWD) -- 62%.\n\n"
                   "Saka is the biggest worry. Want me to find midfielders who are nailed-on to replace him?"),
        ],
        "turns": ["yes"],
        "continuity": _replaces_saka,
        "required_tools": {"find_replacements"},
        "max_tool_calls": 1,
    },
]

# What a reply that lost the thread sounds like ("I don't have anything
# pending from before -- could you tell me what you'd like me to check?").
_LOST_THREAD = ("anything pending", "what you'd like me to", "what would you like me to", "not sure what you")


def run_case(case, llm_config=None):
    """One fresh context, agent, saver and thread per run."""
    ctx = build_context()
    graph = build_eval_agent(ctx, checkpointer=InMemorySaver(), llm_config=llm_config)
    config = thread_config(str(uuid.uuid4()), app="fpl-starts-eval")

    seeded_texts = []
    if case.get("seed"):
        seeded = [HumanMessage(content=text) if kind == "human" else AIMessage(content=text)
                  for kind, text in case["seed"]]
        graph.update_state(config, {"messages": seeded})
        seeded_texts = [text for _, text in case["seed"]]

    turns = []
    try:
        for question in case["turns"]:
            before = len(graph.get_state(config).values.get("messages", []))
            with common.Timer() as timer:
                result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
            new = result["messages"][before:]  # this turn only: the thread holds the rest
            calls, observations, _ = extract_trajectory(new)
            answer = extract_text(new[-1].content) if new[-1].type == "ai" else ""
            input_tokens, output_tokens = common.usage(new)
            turns.append({"question": question, "calls": calls, "observations": observations, "answer": answer,
                          "seconds": timer.seconds, "input_tokens": input_tokens, "output_tokens": output_tokens})
    except Exception as exc:  # noqa: BLE001 -- an API/graph error fails the run, not the whole eval
        return {"error": "{0}: {1}".format(type(exc).__name__, exc), "turns": turns}

    last = turns[-1]
    # Evidence for the last answer: everything the user said or the seeded
    # transcript showed, and every tool result in the conversation.
    evidence = seeded_texts + [t["question"] for t in turns]
    observations = [o["content"] for t in turns for o in t["observations"]]
    lost = [p for p in _LOST_THREAD if p in last["answer"].lower().replace("’", "'")]

    return {
        "error": None,
        "turns": turns,
        "seed": case.get("seed", []),
        "continuity": {"pass": case["continuity"](last["calls"]) and not lost, "lost_thread_phrases": lost},
        "trajectory": scoring.score_trajectory(
            [c["name"] for c in last["calls"]],
            required_tools=case["required_tools"],
            max_tool_calls=case.get("max_tool_calls"),
        ),
        "numbers": scoring.score_numbers(last["answer"], "\n".join(evidence), observations),
        "scope": scoring.score_scope(last["answer"], "normal"),
    }


CHECKS = ("continuity", "trajectory", "numbers", "scope")


def passed(run, key):
    return run["error"] is None and run[key]["pass"]


def print_run(case, run_number, runs, run):
    print("=" * 72)
    print(f"CASE {case['id']}, run {run_number}/{runs}")

    for kind, text in run.get("seed", []):
        print()
        print(f"SEEDED {'USER' if kind == 'human' else 'ASSISTANT'}")
        print(text)

    # Seeded exchanges count as turns too, so "yes" after a seeded offer is turn 2.
    first = sum(kind == "human" for kind, _ in run.get("seed", [])) + 1
    for number, turn in enumerate(run["turns"], start=first):
        print()
        print(f"TURN {number} USER")
        print(turn["question"])
        print(f"TURN {number} TRAJECTORY")
        if not turn["calls"]:
            print("(no tool calls)")
        for call in turn["calls"]:
            print(f"- {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")
        print(f"TURN {number} ANSWER")
        print(turn["answer"])

    if run["error"]:
        print()
        print("ERROR")
        print(run["error"])
        return

    print()
    print("SCORES (last turn)")
    print(f"Conversation continuity: {'PASS' if run['continuity']['pass'] else 'FAIL'}")
    print(f"Trajectory:              {'PASS' if run['trajectory']['pass'] else 'FAIL'}")
    print(f"Numeric faithfulness:    {'PASS' if run['numbers']['pass'] else 'FAIL'}")
    print(f"Scope adherence:         {'PASS' if run['scope']['pass'] else 'FAIL'}")
    if run["continuity"]["lost_thread_phrases"]:
        print(f"Lost-thread phrases: {run['continuity']['lost_thread_phrases']}")
    trajectory = run["trajectory"]
    if trajectory["missing_required"]:
        print(f"Missing required tools: {trajectory['missing_required']}")
    if trajectory["too_many_calls"]:
        print(f"Too many tool calls: {len(trajectory['calls'])}")
    if run["numbers"]["unsupported_percentages"]:
        print(f"Unsupported percentages: {run['numbers']['unsupported_percentages']}")
    if run["numbers"]["unsupported_money"]:
        print(f"Unsupported £ amounts (£m): {run['numbers']['unsupported_money']}")
    for problem in run["scope"]["problems"]:
        print(f"Scope problem: {problem}")


def run_eval(llm_config, runs, verbose=False):
    """Run every conversation `runs` times against `llm_config`'s model,
    printing failed (or, with `verbose`, all) runs and a report; returns the
    summary (see compare_models.py)."""
    common.print_header("MULTI-TURN EVAL", llm_config, runs)
    results = {}
    for case in CASES:
        results[case["id"]] = []
        for run_number in range(1, runs + 1):
            run = run_case(case, llm_config)
            results[case["id"]].append(run)
            if verbose or not all(passed(run, key) for key in CHECKS):
                print_run(case, run_number, runs, run)

    all_runs = [run for case_runs in results.values() for run in case_runs]
    total = len(all_runs)

    def count(key, case_runs=all_runs):
        return sum(passed(run, key) for run in case_runs)

    print("\n" + "=" * 72)
    print(f"CASE PASS RATE ({runs} runs each)")
    print()
    width = max(len(case["id"]) for case in CASES)
    for case in CASES:
        case_runs = results[case["id"]]
        n = len(case_runs)
        print(f"{case['id']:<{width}}  continuity {count('continuity', case_runs)}/{n}  "
              f"trajectory {count('trajectory', case_runs)}/{n}  faithfulness {count('numbers', case_runs)}/{n}  "
              f"scope {count('scope', case_runs)}/{n}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print()
    print(f"{'Runs:':<26}{total:>3}")
    errors = sum(run["error"] is not None for run in all_runs)
    if errors:
        print(f"{'Errored runs:':<26}{errors:>3}")
    for label, key in (("Conversation continuity:", "continuity"), ("Turn-2 trajectory:", "trajectory"),
                       ("Turn-2 faithfulness:", "numbers"), ("Turn-2 scope:", "scope")):
        n = count(key)
        print(f"{label:<26}{n:>3}/{total:<3} {n / total:>5.0%}")
    print()
    perf = common.performance([turn for run in all_runs for turn in run["turns"]])
    common.print_performance(perf, unit="turn")

    return {"config": llm_config, "runs": total, "errors": errors,
            **{key: count(key) / total for key in CHECKS}, **perf}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2,
                        help="times to run each conversation (default 2) -- each makes several API calls")
    parser.add_argument("--verbose", action="store_true", help="print every run, not only the failed ones")
    common.add_llm_arguments(parser)
    args = parser.parse_args()
    run_eval(common.llm_config_from_args(args), args.runs, args.verbose)


if __name__ == "__main__":
    main()
