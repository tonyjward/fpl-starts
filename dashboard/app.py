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

import streamlit as st

import data
import squad
from agent import PRIOR_SEASON, SEASON, build_agent, extract_text, make_current_squad_tool

st.set_page_config(page_title="P(starts) dashboard", layout="wide")
st.title("P(starts) dashboard")

SOURCES = {"Registered snapshot": data.SOURCE_REGISTERED, "Live (frozen model, current inputs)": data.SOURCE_LIVE}
PLAYER_TABLE = ["web_name", "team", "gameweek", "p_start", "availability_status", "last_gw_role",
                "current_season_start_rate", "previous_season_start_rate", "top_positive", "top_negative"]
SQUAD_COLUMNS = ["code", "web_name", "team", "position", "full_name"]
TEAM_ID_HELP = ("**Where do I find my team ID?** Open your FPL **Points** page on "
                "[fantasy.premierleague.com](https://fantasy.premierleague.com) and look at the URL. "
                "Your team ID is the number after `/entry/`. "
                "Example: `fantasy.premierleague.com/entry/1234567/event/1`")


@st.cache_resource
def frozen_model():
    return data.load_frozen_model()


@st.cache_data(ttl=300)
def registered_predictions(season, target_round):
    return data.load_gameweek_predictions(season, target_round, data.SOURCE_REGISTERED)


@st.cache_data(ttl=600)
def bootstrap():
    return data.fetch_bootstrap()


def gameweek_predictions(season, target_round, source):
    if source == data.SOURCE_REGISTERED:
        return registered_predictions(season, target_round)
    return data.load_gameweek_predictions(season, target_round, source, model=frozen_model())


def squad_frame(players):
    import pandas as pd
    return pd.DataFrame([{k: p.get(k) for k in SQUAD_COLUMNS} for p in players])


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

tab_predictions, tab_squad, tab_performance, tab_chat = st.tabs(
    ["Predictions", "My squad", "Performance", "Ask the agent"])

with tab_predictions:
    st.subheader("P(start) for your current squad -- logistic_availability_v1")
    col1, col2 = st.columns(2)
    pred_round = col1.number_input("Gameweek", min_value=1, max_value=38, value=min(last_gw + 1, 38), step=1,
                                   key="pred_round")
    source_label = col2.radio("Source", list(SOURCES), key="pred_source",
                              help="Registered snapshots are the prospective record and are only read. "
                                   "Live applies the frozen model to the current inputs; nothing is "
                                   "fitted or saved.")
    if st.button("Load", key="pred_load") or "predictions" not in state:
        with st.spinner("Loading predictions..."):
            try:
                state.predictions = gameweek_predictions(SEASON, int(pred_round), SOURCES[source_label])
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
                state.predictions = None
                st.error(str(exc))

    if state.get("predictions") is not None:
        predictions, missing = squad.squad_predictions(state.predictions, current_squad)
        meta = predictions.metadata
        st.caption("{0} | {1} | GW{2} | cutoff {3}{4}".format(
            meta["model_id"], meta["source"], meta["gameweek"], meta["prediction_cutoff"],
            "" if meta["predicted_at"] is None else " | predicted {0}{1}".format(
                meta["predicted_at"], " (after deadline)" if meta["generated_after_deadline"] else "")))
        players = data.with_top_factors(predictions).sort_values("p_start", ascending=False)
        st.dataframe(players[PLAYER_TABLE].round(3), use_container_width=True, hide_index=True)
        if missing:
            st.warning("No prediction for: {0}".format(", ".join(squad.describe(p) for p in missing)))

        labels = dict(zip(players["web_name"].fillna(players["code"].astype(str)) + " (" +
                          players["team"].fillna("?") + ")", players["code"]))
        chosen = st.selectbox("Explain a player", list(labels), key="pred_player")
        if chosen:
            code = labels[chosen]
            row = players.set_index("code").loc[code]
            st.markdown("**P(start): {0:.0%}** (logit {1:+.2f} = intercept {2:+.2f} + contributions)".format(
                row["p_start"], row["logit"], meta["intercept"]))
            positive, negative = predictions.top_factors(code, n=5)
            columns = ["description", "raw_value", "contribution"]
            for frame in (positive, negative):  # raw values mix categories and numbers
                frame["raw_value"] = frame["raw_value"].astype(str)
            left, right = st.columns(2)
            left.markdown("Main positive factors")
            left.dataframe(positive[columns].round(3), use_container_width=True, hide_index=True)
            right.markdown("Main negative factors")
            right.dataframe(negative[columns].round(3), use_container_width=True, hide_index=True)

with tab_squad:
    st.subheader("Current squad vs. P(starts)")
    if state.get("predictions") is None:
        st.info("Load predictions in the Predictions tab first.")
    else:
        predictions, _ = squad.squad_predictions(state.predictions, current_squad)
        table = squad.squad_table(current_squad, predictions)

        def _highlight(row):
            if row["is_captain"]:
                return ["background-color: #2d5a2d"] * len(row)
            if row["is_vice_captain"]:
                return ["background-color: #3a3a1f"] * len(row)
            if row["transferred_in"]:
                return ["background-color: #1f3a5a"] * len(row)
            if row["multiplier"] == 0:
                return ["color: #888888"] * len(row)
            return [""] * len(row)
        st.dataframe(table.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)
        st.caption("Green = captain, yellow = vice-captain, blue = transferred in since GW{0}, "
                   "grey = benched in GW{0}.".format(last_gw))

with tab_performance:
    st.subheader("Gameweek performance, logistic_availability_v1 vs baselines")
    target_round = st.number_input("Gameweek", min_value=1, max_value=38, value=last_gw, step=1,
                                    key="perf_round")
    if st.button("Load", key="perf_load"):
        with st.spinner("Scoring..."):
            try:
                reports = data.load_gameweek_comparison(SEASON, PRIOR_SEASON, target_round)
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
                st.error(str(exc))
                reports = {}
        if not reports:
            st.info("No scored predictions found for round {0} yet.".format(target_round))
        for label, df in reports.items():
            st.markdown("**{0}**".format(label))
            st.dataframe(df.round(4), use_container_width=True)

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
