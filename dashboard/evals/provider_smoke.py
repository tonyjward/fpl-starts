"""Live smoke test: can the configured chat model do what the agent needs?

Not "can it say hello" -- the agent needs a model that takes a system
prompt, binds a tool, returns a structured tool call with the right
arguments, and continues from the tool's result. This checks exactly that
with one harmless synthetic tool (no FPL tools, no data), through the same
llm.build_chat_model the app uses. Exits non-zero on any failure.

Run from dashboard/:

    uv run python -m evals.provider_smoke
    uv run python -m evals.provider_smoke --provider openai --model gpt-6-sol
"""

import argparse
import sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent import extract_text
from evals import common
from llm import LLMConfigError, build_chat_model


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two whole numbers and return the sum."""
    return a + b


def smoke(llm_config):
    """A list of problems -- empty when the model passed."""
    problems = []
    model = build_chat_model(llm_config).bind_tools([add_numbers])
    messages = [
        SystemMessage(content="You are a calculator. Always use the add_numbers tool for arithmetic; "
                              "never work a sum out yourself."),
        HumanMessage(content="What is 17 plus 25?"),
    ]

    with common.Timer() as first_timer:
        first = model.invoke(messages)
    calls = first.tool_calls
    print(f"1. Tool call:      {calls}")
    if not calls:
        return problems + ["no structured tool call returned"]
    call = calls[0]
    if call["name"] != "add_numbers":
        problems.append(f"called {call['name']!r}, not 'add_numbers'")
    try:
        if {int(call["args"]["a"]), int(call["args"]["b"])} != {17, 25}:
            problems.append(f"wrong arguments: {call['args']}")
    except (KeyError, TypeError, ValueError):
        problems.append(f"unusable arguments: {call['args']}")
    if not call.get("id"):
        problems.append("tool call has no id, so its result can't be sent back")
        return problems

    # Send back the correct sum whatever the arguments were, so the
    # continuation is tested on its own.
    messages += [first, ToolMessage(content=str(add_numbers.invoke({"a": 17, "b": 25})), tool_call_id=call["id"])]
    with common.Timer() as second_timer:
        second = model.invoke(messages)
    answer = extract_text(second.content)
    print(f"2. Final answer:   {answer.strip()[:200]!r}")
    if second.tool_calls:
        problems.append(f"called tools again after the result: {second.tool_calls}")
    if "42" not in answer:
        problems.append("the final answer doesn't use the tool's result (42)")

    input_tokens, output_tokens = common.usage([first, second])
    print(f"   Latency:        {first_timer.seconds:.1f}s + {second_timer.seconds:.1f}s")
    print(f"   Tokens:         {input_tokens} in, {output_tokens} out (as reported)")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_llm_arguments(parser)
    args = parser.parse_args()
    try:
        llm_config = common.llm_config_from_args(args)
        print(f"PROVIDER SMOKE TEST: {llm_config.label}")
        problems = smoke(llm_config)
    except LLMConfigError as exc:
        problems = [str(exc)]
    except Exception as exc:  # noqa: BLE001 -- an API error is a failed smoke test, reported plainly
        problems = ["{0}: {1}".format(type(exc).__name__, exc)]

    if problems:
        print("FAIL")
        for problem in problems:
            print(f"- {problem}")
        sys.exit(1)
    print("PASS -- tool binding, structured tool call and tool-result continuation all work")


if __name__ == "__main__":
    main()
