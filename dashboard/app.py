"""Streamlit dashboard for the frozen logistic P(starts) model
(logistic_availability_v1), for one FPL manager's current squad.

    uv run streamlit run app.py

The squad is established first (squad.py): team ID -> validated against the
FPL API -> official squad at the end of the last completed gameweek ->
transfers made since then -> effective current squad. Only then is the main
page shown, for the current squad: per-player predictions with their
explanations on the left, and a LangGraph agent chat beside them on the
right.

Read-only against derived.db, predictions/ and the public FPL API -- see
data.py's module docstring. Requires ANTHROPIC_API_KEY (or an `ant auth
login` profile) for the chat only.
"""

import os
import uuid

import pandas as pd
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from fpl_starts import explanation

import data
import squad
import tools
from agent import SEASON, build_agent, extract_text, make_app_tools, thread_config

st.set_page_config(page_title="Who's likely to start?", layout="wide")
st.title("Who's likely to start?")

PLAYER_TABLE = {  # field -> heading
    "full_name": "Player", "team": "Team", "p_start": "Chance of starting",
    "availability_status": "Availability", "last_gw_role": "Last GW",
    "current_season_start_rate": "Start rate (this season)", "previous_season_start_rate": "Start rate (last season)",
}
SQUAD_COLUMNS = {"full_name": "Player", "team": "Team", "position": "Position"}  # field -> heading
AVAILABILITY_LABELS = {
    "available": "Available", "doubtful_75": "Doubtful (75%)", "doubtful_50": "Doubtful (50%)",
    "doubtful_25": "Doubtful (25%)", "injured": "Injured", "suspended": "Suspended",
    "unavailable": "Unavailable", "unknown": "Unknown",
}
LAST_GW_LABELS = {
    "started_60_plus": "Started, 60+ mins", "started_under_60": "Started, under 60 mins",
    "sub_appearance": "Came off the bench", "did_not_play": "Didn't play",
}
PERCENT_COLUMNS = ["Chance of starting", "Start rate (this season)", "Start rate (last season)"]
PERCENT_FORMAT = {c: st.column_config.NumberColumn(format="%d%%") for c in PERCENT_COLUMNS}
TEAM_ID_HELP = ("**Where do I find my team ID?** Open your FPL **Points** page on "
                "[fantasy.premierleague.com](https://fantasy.premierleague.com) and look at the URL. "
                "Your team ID is the number after `/entry/`. "
                "Example: `fantasy.premierleague.com/entry/1234567/event/1`")


# Everything read from derived.db or predictions/ is cached per database
# version: a rebuild (e.g. after a refresh) replaces the file, so the next
# read picks up the new data.
@st.cache_data
def latest_forecast(season, target_round, db_version):
    """The latest registered forecast for `target_round`, read-only."""
    return data.load_gameweek_predictions(season, target_round, data.SOURCE_REGISTERED)


@st.cache_data
def player_universe(db_version):
    """The player list from derived.db."""
    return data.load_player_universe()


@st.cache_data
def player_status(db_version):
    """Each player's latest price, FPL status and news from derived.db."""
    return data.load_player_status()


@st.cache_data
def data_as_of(db_version):
    return data.data_as_of()


# Session keys the tools may change, copied back into the session afterwards:
# a bank the user told the chat, and the forecast reloaded after a refresh.
TOOL_WRITES = ("bank_override", "predictions")


def tools_context():
    """A snapshot of this session and the current data for the tools, built
    here on the script thread. LangGraph runs tool calls in worker threads,
    where Streamlit's session state and caching aren't available, so the
    tools only ever touch this snapshot; `keep_tool_writes` copies what they
    changed back into the session."""
    version = data.db_version()
    return tools.Context(state=dict(st.session_state), universe=player_universe(version),
                         status=player_status(version), data_as_of=data_as_of(version))


def keep_tool_writes(ctx):
    for key in TOOL_WRITES:
        if key in ctx.state:
            st.session_state[key] = ctx.state[key]


def refresh_and_report(ctx):
    """Refresh our FPL data (and the forecast, before the deadline), reload
    the forecast into `ctx`, and describe what changed for the squad. Safe
    on a worker thread: no Streamlit calls."""
    def reload():
        try:
            ctx.state["predictions"] = data.load_gameweek_predictions(SEASON, ctx.state["last_completed_gameweek"] + 1)
        except Exception:  # noqa: BLE001 -- the Predictions tab shows the error
            ctx.state["predictions"] = None
        return tools.Context(state=ctx.state, universe=data.load_player_universe(),
                             status=data.load_player_status(), data_as_of=data.data_as_of())
    return tools.refresh_data(ctx, data.refresh_fpl_data, reload)


def new_agent_thread(state):
    """Start a new conversation: a new opaque thread ID has no checkpoints,
    so the agent's next turn sees only its own question. Memory lives in the
    session agent's InMemorySaver -- lost on a server restart or a new
    browser session, by design for now."""
    state.agent_thread_id = str(uuid.uuid4())
    state.chat_history = []


def presentable(frame, columns):
    """`frame`'s `columns` ({field: heading}) with readable headings and
    values: labels for availability/last-GW codes, rates as percentages."""
    out = frame[list(columns)].rename(columns=columns)
    for heading, labels in (("Availability", AVAILABILITY_LABELS), ("Last GW", LAST_GW_LABELS)):
        if heading in out:
            out[heading] = out[heading].map(labels).fillna(out[heading])
    for heading in PERCENT_COLUMNS:
        if heading in out:
            out[heading] = (out[heading] * 100).round()
    return out


def squad_frame(players):
    return pd.DataFrame([{heading: p.get(k) for k, heading in SQUAD_COLUMNS.items()} for p in players])


# --- onboarding: team -> official squad -> transfers -> current squad -------------------------

state = st.session_state
current_stage = squad.stage(state)

if current_stage == squad.NO_TEAM:
    with st.form("team_id_form"):
        team_id_text = st.text_input("What is your FPL team ID?", placeholder="e.g. 44", key="team_id_input")
        submitted = st.form_submit_button("Continue")
    st.caption(TEAM_ID_HELP)
    if submitted:
        error = squad.submit_team_id(state, team_id_text, data.fetch_team_summary)
        if error:
            st.error(error)
        else:
            st.rerun()
    st.stop()

header, change = st.columns([5, 1])
header.markdown("**{0}**{1} -- team ID {2}".format(
    state.get("team_name") or "Your team", " ({0})".format(state["manager_name"]) if state.get("manager_name") else "",
    state["team_id"]))
if change.button("Change team", key="change_team"):
    squad.change_team(state)
    st.rerun()

try:
    universe = player_universe(data.db_version())
    if current_stage == squad.TEAM_ID_VALID:
        latest_gw = data.last_completed_gameweek()
except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
    st.error("Couldn't load our FPL player data: {0}".format(exc))
    st.stop()

if current_stage == squad.TEAM_ID_VALID:
    error = squad.load_official_squad(state, universe, latest_gw, data.fetch_team_picks)
    if error:
        st.error(error)
        st.stop()

last_gw = state["last_completed_gameweek"]
st.info(squad.FRESHNESS_MESSAGE)
with st.expander("Official squad at the end of GW{0}".format(last_gw),
                 expanded=squad.stage(state) == squad.OFFICIAL_SQUAD_LOADED):
    st.dataframe(squad_frame(state["official_squad"]), use_container_width=True, hide_index=True)

if state.get("transfer_overrides"):
    st.markdown("Transfers since GW{0}: {1}".format(last_gw, "; ".join(
        "{0} -> {1}".format(o["out_name"], o["in_name"]) for o in state["transfer_overrides"])))

with st.form("transfer_form", clear_on_submit=True):
    ready = squad.stage(state) == squad.CURRENT_SQUAD_READY
    message = st.text_input(
        "Any other transfers since GW{0}?".format(last_gw) if ready else
        "Have you made any transfers since GW{0}?".format(last_gw),
        placeholder="e.g. No changes -- or: Player A out for Player B", key="transfer_input")
    sent = st.form_submit_button("Update squad" if ready else "Confirm squad")
if sent:
    error = squad.submit_transfer_message(state, message, universe)
    if error:
        st.warning(error)
    else:
        st.rerun()
if state.get("transfer_overrides") and st.button("Undo transfers", key="undo_transfers"):
    squad.reset_transfers(state)
    st.rerun()

if squad.stage(state) != squad.CURRENT_SQUAD_READY:
    st.stop()

current_squad = state["current_squad"]

# --- predictions for the current squad ----------------------------------------------------

# Full-width predictions, then the conversation below them, with the chat
# input pinned to the bottom of the window (see below) so it's always in view.
predictions_area, chat_area = st.container(), st.container()

with predictions_area:
    next_gw = last_gw + 1
    st.subheader("How likely is each of your players to start in gameweek {0}?".format(next_gw))
    info, button = st.columns([4, 1])
    info.caption("FPL data as of {0}.".format(tools.when(data_as_of(data.db_version()))))
    if button.button("Check for latest FPL news", key="refresh_data",
                     help="Fetches FPL's latest injury news and, before the deadline, updates the forecast. "
                          "Everyone sees the refreshed data; at most once every 30 minutes."):
        with st.spinner("Checking FPL for the latest news..."):
            try:
                ctx = tools_context()
                state.refresh_message = refresh_and_report(ctx)
                keep_tool_writes(ctx)
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
                state.refresh_message = "Couldn't refresh from FPL right now: {0}".format(exc)
        st.rerun()
    if state.get("refresh_message"):
        st.info(state.pop("refresh_message"))
    # Reload whenever derived.db has been rebuilt -- e.g. anyone's refresh
    # registered a new forecast -- not only on this session's first load.
    version = data.db_version()
    if state.get("predictions") is None or state.get("predictions_version") != version:
        try:
            state.predictions = latest_forecast(SEASON, next_gw, version)
            state.predictions_version = version
        except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
            st.error(str(exc))

    if state.get("predictions") is not None:
        predictions, missing = squad.squad_predictions(state.predictions, current_squad)
        meta = predictions.metadata
        st.caption("Latest forecast from {0}, made {1}.".format(meta["model_id"], tools.when(meta["predicted_at"])))
        players = predictions.players.sort_values("p_start", ascending=False)
        players = players.assign(full_name=[
            (universe[c].get("display_name") or universe[c].get("full_name") or w) if c in universe else w
            for c, w in zip(players["code"], players["web_name"])])
        st.dataframe(presentable(players, PLAYER_TABLE), column_config=PERCENT_FORMAT,
                     use_container_width=True, hide_index=True)
        if missing:
            st.warning("No prediction for: {0}".format(", ".join(squad.describe(p) for p in missing)))

        labels = dict(zip(players["full_name"].fillna(players["code"].astype(str)) + " (" +
                          players["team"].fillna("?") + ")", players["code"]))
        chosen = st.selectbox("Explain a player", list(labels), key="pred_player")
        if chosen:
            code = labels[chosen]
            row = players.set_index("code").loc[code]
            st.markdown("**Chance of starting: {0:.0%}**".format(row["p_start"]))
            st.caption("A regular starter (available and playing every week) would be at {0:.0%}. "
                       "The table shows what's holding him back.".format(meta["reference_p_start"]))
            rows = predictions.explain(code)
            if (rows["gap"] < explanation.NEGLIGIBLE_GAP).all():
                st.success("Nothing is holding him back: he's in line with a nailed-on starter.")
            st.dataframe(pd.DataFrame({
                "Factor": rows["label"],
                "What we know": [f[:1].upper() + f[1:] for f in rows["facts"]],
                "Impact": rows["gap"].map(explanation.impact),
                "Chance without this issue": [
                    "{0:.0%}".format(p) if abs(g) >= explanation.NEGLIGIBLE_GAP else "–"
                    for p, g in zip(rows["p_start_if_nailed_on"], rows["gap"])],
            }), use_container_width=True, hide_index=True)

with chat_area:
    st.divider()
    st.subheader("Ask about your squad")
    st.caption("For example: who's at risk in my team? Is Palmer fit? Who could replace Greaves for £5m? "
               "Is our data up to date?")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY isn't set in this environment -- the chat agent "
                   "needs it (or an `ant auth login` profile) to run.")

    if "agent" not in state:
        # The tools read `chat["ctx"]`, a fresh snapshot set before every
        # question (see tools_context), so they always see the current squad,
        # transfers and data without the agent being rebuilt. The checkpointer
        # holds only this session's conversation (one per session agent, never
        # shared between users) -- the squad and data never come from it.
        chat = state.chat = {}
        state.agent = build_agent(app_tools=make_app_tools(
            lambda: squad.current_squad_report(chat["ctx"].state), lambda: chat["ctx"],
            lambda: refresh_and_report(chat["ctx"])), checkpointer=InMemorySaver())
    if "agent_thread_id" not in state:
        new_agent_thread(state)
    if "chat_history" not in state:
        state.chat_history = []
    new_chat_slot = st.container()  # filled at the end of the run -- see below

    # The conversation so far; a new question and answer are written into
    # this container too, so they appear in order below the earlier ones.
    conversation = st.container()
    with conversation:
        for message in state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Called at the top level of the page (not inside a container, column or tab),
# st.chat_input is pinned to the bottom of the window -- always visible,
# however far down the user has scrolled.
question = st.chat_input("e.g. \"Who's at risk in my squad?\" or \"Is Palmer fit?\"")
if question:
    state.chat_history.append({"role": "user", "content": question})
    with conversation:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                version_before = data.db_version()
                state.chat["ctx"] = tools_context()
                # Only the new question: earlier turns come from the
                # checkpointer, so resending chat_history would duplicate them.
                result = state.agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=thread_config(state.agent_thread_id, app="fpl-starts", season=SEASON,
                                         gameweek=state["last_completed_gameweek"] + 1))
                keep_tool_writes(state.chat["ctx"])
                answer = extract_text(result["messages"][-1].content)
            st.markdown(answer)
    state.chat_history.append({"role": "assistant", "content": answer})
    if data.db_version() != version_before:
        # A refresh rebuilt the data after the predictions were drawn on this
        # run -- draw the page again so they show the new forecast.
        st.rerun()

# Drawn last, so it appears as soon as the first answer is in.
if state.chat_history and new_chat_slot.button("New chat", key="new_chat"):
    new_agent_thread(state)
    st.rerun()
