"""LangGraph agent: explains a gameweek's P(starts) numbers, or one
manager's squad against them, in conversation -- never estimates or
invents a probability itself (same discipline as the agent challenger in
../../fpl-starts/src/fpl_starts/agent/predict.py: every number it talks
about comes from a tool call reading real derived.db/API data, not from
the model).

Read-only by construction: every tool below wraps a data.py function, and
data.py never writes to either derived.db or calls anything but public,
unauthenticated FPL API endpoints. No pipeline step, no prediction, no
archive write is reachable from this agent.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import data

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"
MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You explain Fantasy Premier League P(starts) predictions \
and gameweek performance to the person you're talking to. You have two \
tools: one reads how every prediction arm scored for a given gameweek \
(Brier score and accuracy, stratified by Core/Rotation/Marginal/Deep -- \
lower Brier is better, and never quote a single pool-wide number as if it \
were the whole story, since Deep players are trivially predictable and \
dominate any pool average); the other, given an FPL manager's team ID and \
a gameweek, pulls their actual 15-player squad and every model's p_start \
for each of those players.

Never estimate a probability or a score yourself -- every number you state \
must come from a tool call in this conversation, not from your own \
knowledge or a guess. If a tool returns no data for a round (not played \
yet, or not archived), say so plainly rather than filling the gap with a \
plausible-sounding number.

When explaining a squad: point out anyone whose P(starts) is notably low \
given their squad position (especially the captain/vice-captain, or a \
starting XI slot rather than the bench), and note when different model \
arms disagree meaningfully about the same player rather than picking one \
number silently. Keep answers grounded in the actual returned data, and \
concise -- this is a conversation, not a report."""


@tool
def get_gameweek_report(target_round: int) -> str:
    """Stratified Brier score and accuracy for every P(starts) model arm in
    `target_round`, across both repos, against three baselines (persistence,
    season-rate, constant 0.9). Use this to answer "how did the model do in
    gameweek N" or "which arm performed best". Returns a plain-text table,
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
    a team on the FPL site), with every archived model arm's P(starts) for
    each player, and which one is captain/vice-captain. Use this to answer
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


def build_agent():
    llm = ChatAnthropic(model=MODEL, max_tokens=8000)
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
    print(result["messages"][-1].content)


if __name__ == "__main__":
    _main()
