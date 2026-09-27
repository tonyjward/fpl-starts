"""Streamlit dashboard for the frozen logistic P(starts) model
(logistic_availability_v1): per-player predictions with their explanations,
gameweek performance, one manager's squad against those predictions, and a
LangGraph agent chat for explaining either in conversation.

    uv run streamlit run app.py

Read-only against derived.db, predictions/ and the public FPL API -- see
data.py's module docstring. Requires ANTHROPIC_API_KEY (or an `ant auth
login` profile) for the chat tab only; the performance/squad tabs work
without it.
"""

import os

import pandas as pd
import streamlit as st

import data
from agent import PRIOR_SEASON, SEASON, build_agent, extract_text

st.set_page_config(page_title="P(starts) dashboard", layout="wide")
st.title("P(starts) dashboard")

SOURCES = {"Registered snapshot": data.SOURCE_REGISTERED, "Live (frozen model, current inputs)": data.SOURCE_LIVE}
PLAYER_TABLE = ["web_name", "team", "gameweek", "p_start", "availability_status", "last_gw_role",
                "current_season_start_rate", "previous_season_start_rate", "top_positive", "top_negative"]


@st.cache_resource
def frozen_model():
    return data.load_frozen_model()


@st.cache_data(ttl=300)
def registered_predictions(season, target_round):
    return data.load_gameweek_predictions(season, target_round, data.SOURCE_REGISTERED)


def gameweek_predictions(season, target_round, source):
    if source == data.SOURCE_REGISTERED:
        return registered_predictions(season, target_round)
    return data.load_gameweek_predictions(season, target_round, source, model=frozen_model())


tab_predictions, tab_performance, tab_squad, tab_chat = st.tabs(
    ["Predictions", "Performance", "My squad", "Ask the agent"])

with tab_predictions:
    st.subheader("P(start) by player -- logistic_availability_v1")
    col1, col2 = st.columns(2)
    pred_round = col1.number_input("Gameweek", min_value=1, max_value=38, value=6, step=1, key="pred_round")
    source_label = col2.radio("Source", list(SOURCES), key="pred_source",
                              help="Registered snapshots are the prospective record and are only read. "
                                   "Live applies the frozen model to the current inputs; nothing is "
                                   "fitted or saved.")
    if st.button("Load", key="pred_load"):
        with st.spinner("Loading predictions..."):
            try:
                st.session_state.predictions = gameweek_predictions(SEASON, int(pred_round), SOURCES[source_label])
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
                st.session_state.pop("predictions", None)
                st.error(str(exc))

    predictions = st.session_state.get("predictions")
    if predictions is not None:
        meta = predictions.metadata
        st.caption("{0} | {1} | GW{2} | cutoff {3}{4}".format(
            meta["model_id"], meta["source"], meta["gameweek"], meta["prediction_cutoff"],
            "" if meta["predicted_at"] is None else " | predicted {0}{1}".format(
                meta["predicted_at"], " (after deadline)" if meta["generated_after_deadline"] else "")))
        players = data.with_top_factors(predictions).sort_values("p_start", ascending=False)
        st.dataframe(players[PLAYER_TABLE].round(3), use_container_width=True, hide_index=True)

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
            left, right = st.columns(2)
            left.markdown("Main positive factors")
            for frame in (positive, negative):  # raw values mix categories and numbers
                frame["raw_value"] = frame["raw_value"].astype(str)
            left.dataframe(positive[columns].round(3), use_container_width=True, hide_index=True)
            right.markdown("Main negative factors")
            right.dataframe(negative[columns].round(3), use_container_width=True, hide_index=True)

with tab_performance:
    st.subheader("Gameweek performance, logistic_availability_v1 vs baselines")
    target_round = st.number_input("Gameweek", min_value=1, max_value=38, value=3, step=1,
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

with tab_squad:
    st.subheader("My squad vs. P(starts)")
    col1, col2 = st.columns(2)
    team_id = col1.number_input("FPL team ID", min_value=1, value=1, step=1, key="squad_team_id")
    event = col2.number_input("Gameweek", min_value=1, max_value=38, value=3, step=1,
                               key="squad_event")
    st.caption("P(start) from the registered logistic_availability_v1 snapshot for that gameweek.")
    if st.button("Load squad", key="squad_load"):
        with st.spinner("Fetching squad and predictions..."):
            try:
                squad = data.load_squad_predictions(team_id, event, SEASON, event)
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not a crash
                st.error(str(exc))
                squad = pd.DataFrame()
        if not squad.empty:
            def _highlight(row):
                if row["is_captain"]:
                    return ["background-color: #2d5a2d"] * len(row)
                if row["is_vice_captain"]:
                    return ["background-color: #3a3a1f"] * len(row)
                if row["multiplier"] == 0:
                    return ["color: #888888"] * len(row)
                return [""] * len(row)
            st.dataframe(squad.style.apply(_highlight, axis=1), use_container_width=True)
            st.caption("Green = captain, yellow = vice-captain, grey = benched this gameweek.")

with tab_chat:
    st.subheader("Ask about a gameweek or your squad")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY isn't set in this environment -- the chat agent "
                   "needs it (or an `ant auth login` profile) to run.")

    if "agent" not in st.session_state:
        st.session_state.agent = build_agent()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("e.g. \"How did gameweek 3 go?\" or \"Explain team 1's squad for gw3\"")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": question}]}
                )
                answer = extract_text(result["messages"][-1].content)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
