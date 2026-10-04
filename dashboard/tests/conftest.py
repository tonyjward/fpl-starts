"""AppTests run the real app.py with the FPL API and P(start) loading
replaced by the synthetic payloads in fakes.py. Run from dashboard/:

    uv run pytest
"""

import os
import sys

import pytest

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, DASHBOARD_DIR)
sys.path.insert(0, os.path.dirname(__file__))

import data  # noqa: E402
import fakes  # noqa: E402


@pytest.fixture
def fake_fpl(monkeypatch):
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    # The app reads the LLM settings; pin them so a real .env can't change
    # what these tests see (the model itself is faked -- see scripted_llm).
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setattr(data, "load_player_universe", fakes.universe)
    monkeypatch.setattr(data, "last_completed_gameweek", lambda season=None: fakes.LAST_COMPLETED_GW)
    monkeypatch.setattr(data, "db_version", lambda: 0)
    monkeypatch.setattr(data, "load_player_status", fakes.status)
    monkeypatch.setattr(data, "data_as_of", lambda: fakes.DATA_AS_OF)
    monkeypatch.setattr(data, "fetch_team_summary", fakes.fetch_team_summary)
    monkeypatch.setattr(data, "fetch_team_picks", fakes.fetch_team_picks)
    requested = []

    def load_gameweek_predictions(season, target_round, source=data.SOURCE_REGISTERED, model=None):
        requested.append((season, target_round, source))
        return fakes.predictions(target_round, source)
    monkeypatch.setattr(data, "load_gameweek_predictions", load_gameweek_predictions)
    return requested


def scripted_chat_model(tool_name, tool_args):
    """A fake LangChain chat model -- no API calls -- that first asks for one
    tool call, then answers with the text of the tool's result. LangGraph
    runs that tool call in a worker thread, exactly as with the real model."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class Scripted(BaseChatModel):
        calls: list = Field(default_factory=list)

        @property
        def _llm_type(self):
            return "scripted-fake"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append(list(messages))
            last = messages[-1]
            if last.type == "tool":
                reply = AIMessage(content="From the tool: " + str(last.content))
            else:
                reply = AIMessage(content="", tool_calls=[{"name": tool_name, "args": tool_args, "id": "call-1"}])
            return ChatResult(generations=[ChatGeneration(message=reply)])

    return Scripted()


@pytest.fixture
def scripted_llm(monkeypatch):
    """Make the app's agent use a scripted fake model: call
    `scripted_llm(tool_name, args)` before the first chat question."""
    import agent

    holder = {}

    def configure(tool_name, tool_args=None):
        holder["llm"] = scripted_chat_model(tool_name, tool_args or {})
        monkeypatch.setattr(agent, "build_chat_model", lambda config=None: holder["llm"])
        return holder["llm"]
    return configure
