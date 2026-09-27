"""LangGraph agent: explains a gameweek's P(starts) numbers, or one
manager's squad against them, in conversation -- never estimates or
invents a probability itself: every number it talks about comes from a
tool call reading the frozen logistic model's registered predictions,
derived.db or the public FPL API, not from the LLM.

Read-only by construction: every tool below wraps a data.py function, and
data.py never writes to derived.db or predictions/, or calls anything but
public, unauthenticated FPL API endpoints. No pipeline step, no prediction, no
archive write is reachable from this agent.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import data

# load environment variables
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"
MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You explain Fantasy Premier League P(starts) predictions \
and gameweek performance to the person you're talking to. The P(starts) \
model is logistic_availability_v1, a frozen, interpretable logistic \
regression. You have two tools: one reads how the model scored for a given \
gameweek (Brier score and accuracy, stratified by Core/Rotation/Marginal/Deep -- \
lower Brier is better, and never quote a single pool-wide number as if it \
were the whole story, since Deep players are trivially predictable and \
dominate any pool average); the other, given an FPL manager's team ID and \
a gameweek, pulls their actual 15-player squad with the model's p_start, \
availability status, last-gameweek role and start rates for each player.

Never estimate a probability or a score yourself -- every number you state \
must come from a tool call in this conversation, not from your own \
knowledge or a guess. If a tool returns no data for a round (not played \
yet, or not archived), say so plainly rather than filling the gap with a \
plausible-sounding number.

When explaining a squad: point out anyone whose P(starts) is notably low \
given their squad position (especially the captain/vice-captain, or a \
starting XI slot rather than the bench), and use the availability status, \
last-gameweek role and start rates to say why. Keep answers grounded in the actual returned data, and \
concise -- this is a conversation, not a report."""


@tool
def get_gameweek_report(target_round: int) -> str:
    """Stratified Brier score and accuracy for the logistic P(starts) model
    in `target_round`, against three baselines (persistence, season-rate,
    constant 0.9). Use this to answer "how did the model do in gameweek N"
    or "did it beat persistence". Returns a plain-text table,
    or a message saying no data exists yet if the round hasn't been scored.
    """
    reports = data.load_gameweek_comparison(SEASON, PRIOR_SEASON, target_round)
    if not reports:
        return ("No scored predictions found for {0} round {1} -- either it hasn't "
                 "been played yet, or the outcome hasn't been archived.").format(
                     SEASON, target_round)
    parts = []
    for label, df in reports.items():
        parts.append("--- {0} ---\n{1}".format(label, df.round(4).to_string()))
    return "\n\n".join(parts)


@tool
def get_team_squad_predictions(team_id: int, event: int) -> str:
    """One FPL manager's actual 15-player squad for gameweek `event`
    (`team_id` is their public FPL entry ID, e.g. from the URL when viewing
    a team on the FPL site), with the logistic model's registered P(starts),
    availability status, last-gameweek role and start rates for each player,
    and which one is captain/vice-captain. Use this to answer
    "explain my team" or "should I be worried about my captain's P(starts)".
    Returns a plain-text table, or an error message if the team ID or
    gameweek is invalid, or no predictions are archived for that round yet.
    """
    try:
        squad = data.load_squad_predictions(team_id, event, SEASON, event)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the model as a tool
        # result, not raised, so it can explain the problem instead of the
        # conversation just erroring out (invalid team_id, gameweek not
        # played for this manager yet, no predictions archived, etc.)
        return "Could not load that team/gameweek: {0}".format(exc)
    if squad.empty:
        return "No picks found for team {0}, gameweek {1}.".format(team_id, event)
    return squad.round(4).to_string(index=False)


def extract_text(content):
    """The plain-text answer from one message's `.content` -- Opus 5 thinks
    by default, so `content` is a list of blocks (thinking + text), not a
    bare string; printing/rendering the list directly (confirmed live)
    dumps the thinking block's raw signature next to the real answer.
    Some LangChain integrations do flatten to a plain string, so handle
    both rather than assume the list shape.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def build_agent():
    # An API key that isn't scoped to a single workspace needs this header on
    # every request (confirmed live -- omitting it 400s), a key that *is*
    # scoped doesn't need or accept it being wrong, so only send it when set.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    llm = ChatAnthropic(model=MODEL, max_tokens=8000, default_headers=default_headers)
    return create_react_agent(llm, [get_gameweek_report, get_team_squad_predictions],
                               prompt=SYSTEM_PROMPT)


def _main():
    """Quick CLI smoke test: `uv run python agent.py "your question"` --
    the Streamlit app (app.py) is the real interface.
    """
    import sys

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set -- see `ant auth login` or export it.")

    question = " ".join(sys.argv[1:]) or "Explain gameweek 3's results."
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(extract_text(result["messages"][-1].content))


if __name__ == "__main__":
    _main()
