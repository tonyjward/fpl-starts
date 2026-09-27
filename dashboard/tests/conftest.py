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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setattr(data, "load_player_universe", fakes.universe)
    monkeypatch.setattr(data, "last_completed_gameweek", lambda season=None: fakes.LAST_COMPLETED_GW)
    monkeypatch.setattr(data, "db_version", lambda: 0)
    monkeypatch.setattr(data, "fetch_team_summary", fakes.fetch_team_summary)
    monkeypatch.setattr(data, "fetch_team_picks", fakes.fetch_team_picks)
    requested = []

    def load_gameweek_predictions(season, target_round, source=data.SOURCE_REGISTERED, model=None):
        requested.append((season, target_round, source))
        return fakes.predictions(target_round, source)
    monkeypatch.setattr(data, "load_gameweek_predictions", load_gameweek_predictions)
    return requested
