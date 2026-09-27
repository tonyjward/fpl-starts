"""Streamlit dashboard for the frozen logistic P(starts) model
(logistic_availability_v1), for one FPL manager's current squad.

    uv run streamlit run app.py

The squad is established first (squad.py): team ID -> validated against the
FPL API -> official squad at the end of the last completed gameweek ->
transfers made since then -> effective current squad. Only then are the
tabs shown: per-player predictions with their explanations, the squad view,
gameweek performance, and a LangGraph agent chat -- all for the current
squad.

Read-only against derived.db, predictions/ and the public FPL API -- see
data.py's module docstring. Requires ANTHROPIC_API_KEY (or an `ant auth
login` profile) for the chat tab only.
"""

import os

import pandas as pd
import streamlit as st

from fpl_starts import explanation

import data
import squad
from agent import SEASON, build_agent, extract_text, make_current_squad_tool

st.set_page_config(page_title="Who's likely to start?", layout="wide")
st.title("Who's likely to start?")

PLAYER_TABLE = {  # field -> heading
    "web_name": "Player", "team": "Team", "p_start": "Chance of starting",
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


@st.cache_data(ttl=300)
def latest_forecast(season, target_round):
    """The latest registered forecast for `target_round`, read-only."""
    return data.load_gameweek_predictions(season, target_round, data.SOURCE_REGISTERED)


@st.cache_data(ttl=600)
def bootstrap():
    return data.fetch_bootstrap()


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
        team_id_text = st.text_input("What is your FPL team ID?", placeholder="e.g. 1234567", key="team_id_input")
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
    fpl_bootstrap = bootstrap()
except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
    st.error("Couldn't load the FPL player list: {0}".format(exc))
    st.stop()
universe = squad.player_universe(fpl_bootstrap)

if current_stage == squad.TEAM_ID_VALID:
    error = squad.load_official_squad(state, fpl_bootstrap, data.fetch_team_picks)
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

tab_predictions, tab_chat = st.tabs(["Predictions", "Ask the agent"])

with tab_predictions:
    next_gw = last_gw + 1
    st.subheader("How likely is each of your players to start in gameweek {0}?".format(next_gw))
    if state.get("predictions") is None:
        try:
            state.predictions = latest_forecast(SEASON, next_gw)
        except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
            st.error(str(exc))

    if state.get("predictions") is not None:
        predictions, missing = squad.squad_predictions(state.predictions, current_squad)
        meta = predictions.metadata
        st.caption("Latest forecast from {0}, made {1}.".format(
            meta["model_id"], pd.Timestamp(meta["predicted_at"]).strftime("%d %b %Y %H:%M UTC")))
        players = predictions.players.sort_values("p_start", ascending=False)
        st.dataframe(presentable(players, PLAYER_TABLE), column_config=PERCENT_FORMAT,
                     use_container_width=True, hide_index=True)
        if missing:
            st.warning("No prediction for: {0}".format(", ".join(squad.describe(p) for p in missing)))

        labels = dict(zip(players["web_name"].fillna(players["code"].astype(str)) + " (" +
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

with tab_chat:
    st.subheader("Ask about a gameweek or your squad")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY isn't set in this environment -- the chat agent "
                   "needs it (or an `ant auth login` profile) to run.")

    if "agent" not in state:
        # The squad tool reads session state when called, so it always sees
        # the current squad (and a changed team) without rebuilding the agent.
        state.agent = build_agent(squad_tool=make_current_squad_tool(lambda: squad.current_squad_report(state)))
    if "chat_history" not in state:
        state.chat_history = []

    for message in state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("e.g. \"How did gameweek 5 go?\" or \"Should I worry about my captain?\"")
    if question:
        state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = state.agent.invoke({"messages": [{"role": "user", "content": question}]})
                answer = extract_text(result["messages"][-1].content)
            st.markdown(answer)
        state.chat_history.append({"role": "assistant", "content": answer})
