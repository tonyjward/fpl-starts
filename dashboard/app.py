"""Streamlit dashboard: P(starts) performance across both repos' arms, one
manager's squad against those predictions, and a LangGraph agent chat for
explaining either in conversation.

    uv run streamlit run app.py

Read-only against both derived.db files and the public FPL API -- see
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

tab_performance, tab_squad, tab_chat = st.tabs(["Performance", "My squad", "Ask the agent"])

with tab_performance:
    st.subheader("Gameweek performance, every arm")
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
