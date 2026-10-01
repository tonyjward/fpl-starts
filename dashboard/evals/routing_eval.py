"""Live evaluation of the agent's first routing decision.

Uses the real production system prompt, model and tool schemas, but does
NOT execute any tools. This evaluates:

- tool selection
- argument extraction

Run from dashboard/:

    uv run python -m evals.routing_eval
"""

import os
from math import isclose

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
    MODEL,
    SYSTEM_PROMPT,
    get_gameweek_report,
    make_app_tools,
)


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
        "expected_tool": "player_news",
        "expected_args": {"name": "Palmer"},
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


def build_router():
    """Build the same Claude + tool interface used by the real app.

    The callback bodies are deliberately harmless: this script never
    executes the tools. We only ask Claude which tool it would call.
    """

    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = (
        {"anthropic-workspace-id": workspace_id}
        if workspace_id
        else None
    )

    llm = ChatAnthropic(
        model=MODEL,
        max_tokens=1000,
        default_headers=default_headers,
    )

    app_tools = make_app_tools(
        get_report=lambda: "stub",
        get_context=lambda: None,
        refresh_and_report=lambda: "stub",
    )

    tools = [get_gameweek_report] + app_tools

    return llm.bind_tools(tools)


def value_matches(actual, expected):
    if isinstance(expected, float):
        try:
            return isclose(float(actual), expected, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False

    return actual == expected


def expected_args_match(actual, expected):
    """Expected args are a subset.

    Claude is allowed to explicitly send optional defaults that we did
    not specify in the golden case.
    """

    return all(
        key in actual and value_matches(actual[key], value)
        for key, value in expected.items()
    )


def main():
    router = build_router()

    tool_correct = 0
    args_correct = 0

    for number, case in enumerate(CASES, start=1):
        response = router.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=case["question"]),
            ]
        )

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

        tool_correct += int(tool_ok)
        args_correct += int(args_ok)

        print("=" * 72)
        print(f"CASE {number}")
        print(f"Question:       {case['question']}")
        print(f"Expected tool:  {case['expected_tool']}")
        print(f"Actual tool:    {actual_tool}")
        print(f"Expected args:  {case['expected_args']}")
        print(f"Actual args:    {actual_args}")
        print(f"Tool routing:   {'PASS' if tool_ok else 'FAIL'}")
        print(f"Arguments:      {'PASS' if args_ok else 'FAIL'}")

        if len(calls) > 1:
            print(f"Extra calls:    {calls[1:]}")

    total = len(CASES)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"Tool routing:     {tool_correct}/{total} ({tool_correct / total:.1%})")
    print(f"Argument routing: {args_correct}/{total} ({args_correct / total:.1%})")


if __name__ == "__main__":
    main()