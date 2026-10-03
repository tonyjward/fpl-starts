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

SYSTEM_PROMPT = """You help a Fantasy Premier League manager with one \
question: who is likely to start. The predictions come from \
logistic_availability_v1, a frozen, interpretable logistic regression, and \
each is explained against a regular starter (available and starting every \
week). Your tools cover: the manager's current squad (their official squad \
from the last completed gameweek plus the transfers they've told the app \
about); any player's chance of starting, why, and their FPL news; starting-XI \
risks with legal bench swaps; replacements likely to start, by position and \
budget; FPL's injury/suspension news across the squad and whether it \
changed since the forecast; \
refreshing our FPL data (and, before the deadline, the forecast); and how \
the model scored in past gameweeks (Brier score and accuracy by \
Core/Rotation/Marginal/Deep -- lower is better; never quote the pool-wide \
number alone, since Deep players are trivially predictable).

You can only speak to chance of starting. You have no model of points, \
form, fixtures' difficulty or value: if asked who to captain, who will score \
or which transfer is "best", say plainly that you can tell them who's likely \
to start, not who'll score -- then offer what you can (e.g. which options are \
nailed-on). Budgets: FPL doesn't publish selling prices, so the bank is an \
estimate unless the user tells you theirs; if they do, pass it to the \
replacements tool. For a replacement question ("who can replace X?"), call \
the replacements tool alone -- it already looks up X's position and price, \
so don't also call the player tool on X. Every tool answer states when our \
FPL data is from; \
mention it when news or prices matter. If the user thinks our news is out \
of date, use the refresh tool (it only works before the deadline, and not \
more than every 30 minutes).

Never estimate a probability or a score yourself -- every number you state \
must come from a tool call in this conversation, not from your own \
knowledge or a guess. If a tool returns no data for a round (not played \
yet, or not archived), say so plainly rather than filling the gap with a \
plausible-sounding number.

When explaining a squad: point out anyone whose P(starts) is notably low \
given their squad position (especially the captain/vice-captain, or a \
starting XI slot rather than the bench), and use the availability status, \
last-gameweek role and start rates to say why. Talk to the user in plain \
English: say "chance of starting" (as a percentage), never "P(start)", \
"p_start", "logit" or other model jargon. Keep answers grounded in the actual returned data, and \
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


def make_app_tools(get_report, get_context, refresh_and_report):
    """The app's session tools. Each calls back into the session when the
    model uses it -- `get_report()` for the squad report, `get_context()`
    for a tools.Context on the current data, `refresh_and_report()` for a
    refresh -- so they always see the current squad and data without the
    agent being rebuilt."""
    import tools as session_tools

    @tool
    def get_my_current_squad_predictions() -> str:
        """The user's current FPL squad (their official squad from the last
        completed gameweek, with the transfers they've made since applied),
        with the logistic model's P(starts), availability status,
        last-gameweek role, start rates and the largest positive/negative
        factors for each player. Use this for any question about "my team"
        or "my squad". Returns a plain-text table, or a message saying the
        squad or its predictions aren't ready yet.
        """
        return get_report()

    @tool
    def explain_player(name: str) -> str:
        """Everything about one player (in the user's squad or not): chance
        of starting in the upcoming gameweek, what's holding him back
        compared with a regular starter, his price, and FPL's injury/
        suspension status and news (and whether it changed since the
        forecast). Use for any single-player question, including "is X fit?"
        or "is X injured?". `name` as the user said it (e.g. "Cole Palmer",
        "Bruno"). Asks for clarification if the name is ambiguous -- relay
        that question to the user.
        """
        return session_tools.explain_player(get_context(), name)

    @tool
    def squad_risks(threshold: float = 0.75) -> str:
        """Starting-XI players whose chance of starting is below `threshold`
        (0-1, default 0.75), why, and bench players who are more likely to
        start and could come in while keeping a legal formation. Use for
        "who should I worry about?" or "should I change my bench?".
        """
        return session_tools.squad_risks(get_context(), threshold)

    @tool
    def find_replacements(replacing: str = None, position: str = None, max_price: float = None,
                          bank: float = None, min_chance: float = 0.75) -> str:
        """Players likely to start (chance >= `min_chance`, 0-1) who aren't
        in the user's squad. Give `replacing` (a squad player's name) to use
        his position and budget (his price + the bank); or `position`
        ("goalkeeper", "defender", "midfielder", "forward") and optionally
        `max_price` in £m. Pass `bank` (£m) if the user tells you how much
        they have -- it's remembered. Respects the 3-per-club limit. Looks
        up the outgoing player itself, so there's no need to call
        explain_player on him first.
        """
        return session_tools.find_replacements(get_context(), replacing=replacing, position=position,
                                                max_price=max_price, bank=bank, min_chance=min_chance)

    @tool
    def player_news() -> str:
        """FPL's injury/suspension news across the user's whole squad: every
        squad player with news or whose status has changed since the
        forecast was made. Use for "any injury news in my team?". For one
        player's news, use explain_player instead.
        """
        return session_tools.player_news(get_context())

    @tool
    def refresh_fpl_data() -> str:
        """Fetch the latest FPL data (status, news, prices) and, before the
        upcoming gameweek's deadline, update the forecast; then report what
        changed for the user's squad. Use when the user asks whether their
        information is up to date or suspects news is missing. Refused after
        the deadline or within 30 minutes of the last refresh.
        """
        return refresh_and_report()

    return [get_my_current_squad_predictions, explain_player, squad_risks, find_replacements, player_news,
            refresh_fpl_data]


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


def build_agent(app_tools=None, checkpointer=None):
    """`app_tools` (from make_app_tools) replace the command-line team-ID
    squad tool -- the app passes them so chat works on the user's current
    squad and session.

    `checkpointer` (a LangGraph checkpoint saver) gives the agent memory
    across turns: invoked with the same `configurable.thread_id` (see
    thread_config), each turn sees the conversation so far. Without one, as
    in the CLI and the evals, every invoke starts from nothing. The app
    passes a new InMemorySaver per Streamlit session -- never share one
    between users."""
    # An API key that isn't scoped to a single workspace needs this header on
    # every request (confirmed live -- omitting it 400s), a key that *is*
    # scoped doesn't need or accept it being wrong, so only send it when set.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    llm = ChatAnthropic(model=MODEL, max_tokens=8000, default_headers=default_headers)
    tools = [get_gameweek_report] + list(app_tools or [get_team_squad_predictions])
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


def thread_config(thread_id, **metadata):
    """The invoke config for one conversation. The same opaque ID serves two
    separate purposes: `configurable.thread_id` is the LangGraph checkpointer's
    key for this conversation's messages (behaviour: "yes" knows what was
    just offered), and `metadata.thread_id` is what LangSmith groups traces
    into a thread by (observability only -- LangSmith is never the source of
    the agent's memory). Keep `metadata` small and non-identifying."""
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id, **metadata},
    }


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
