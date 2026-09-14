"""Tests for agent/predict.py: the manual tool-use loop, quote verification,
and the classify-then-lookup end-to-end flow. No real network or Anthropic
API calls anywhere -- llm_client/search_web/fetch_page_text are all fakes.
"""

import json
import sqlite3

import pandas as pd
import pytest

from fpl_starts.agent.predict import (
    load_club_roster,
    predict_club_agent,
    run_agent_loop,
    verify_classifications,
)
from fpl_starts.agent.tools import ToolBudget

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"


def make_fake_fetch(responses):
    def fetch(path):
        if path not in responses:
            raise AssertionError("unexpected path: {0}".format(path))
        return responses[path]
    return fetch


def merged_gw_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def players_raw_csv(id_code_pairs):
    return pd.DataFrame(
        [{"id": i, "code": c} for i, c in id_code_pairs]
    ).to_csv(index=False).encode("utf-8")


def make_derived_db(path, players, teams_rows, gameweek_rows):
    """players: (code, web_name, team_code). teams_rows: (code, name)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE players (code INTEGER PRIMARY KEY, web_name TEXT, "
        "team_code INTEGER)"
    )
    conn.executemany("INSERT INTO players VALUES (?, ?, ?)", players)
    conn.execute("CREATE TABLE teams (code INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO teams VALUES (?, ?)", teams_rows)
    conn.execute(
        "CREATE TABLE player_gameweek_stats (code, season, round, starts, "
        "team_code INTEGER)"
    )
    conn.executemany(
        "INSERT INTO player_gameweek_stats (code, season, round, starts) "
        "VALUES (?, ?, ?, ?)", gameweek_rows
    )
    conn.execute(
        "CREATE TABLE player_availability_snapshots "
        "(code, season, fetched_at, next_gw, status, chance_of_playing_next_round)"
    )
    conn.execute(
        "CREATE TABLE predictions (code, season, target_round, model_version, method)"
    )
    conn.commit()
    return conn


class FakeBlock(object):
    def __init__(self, text):
        self.text = text


class FakeMessage(object):
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeMessagesResource(object):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(self._responses.pop(0))


class FakeAnthropicClient(object):
    def __init__(self, responses):
        self.messages = FakeMessagesResource(responses)


def fake_roster():
    return pd.DataFrame({
        "code": [1001, 1002],
        "web_name": ["Alice", "Bob"],
        "p_start": [0.5, 0.6],
        "cold_start": [False, False],
        "n_observed": [10, 10],
        "method": ["cal_rolling_xseason", "cal_rolling_xseason"],
    })


# --------------------------------------------------------------------------
# load_club_roster
# --------------------------------------------------------------------------


def test_load_club_roster_filters_to_one_club(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[(1001, "Alice", 1), (1002, "Bob", 2)],
        teams_rows=[(1, "Leeds"), (2, "Newcastle")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1002, SEASON, 1, 1)],
    )
    roster = load_club_roster(conn, SEASON, PRIOR_SEASON, 2, "Leeds", fetch=fetch)
    conn.close()
    assert list(roster["code"]) == [1001]


# --------------------------------------------------------------------------
# run_agent_loop
# --------------------------------------------------------------------------


def test_run_agent_loop_happy_path_search_then_fetch_then_final_answer():
    quote = "Alice will start on Saturday, the manager confirmed."
    responses = [
        json.dumps({"action": "search_web", "query": "Leeds team news"}),
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_starting",
             "quote": quote, "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)
    budget = ToolBudget(max_searches=3, max_fetches=2)

    def fake_search(query):
        return [{"title": "t", "url": "https://a", "snippet": "s"}]

    def fake_fetch(url):
        return quote

    classifications, pages = run_agent_loop(
        client, "fake-model", "Leeds", fake_roster(), budget,
        search_web=fake_search, fetch_page_text=fake_fetch,
    )

    assert classifications == [
        {"code": 1001, "category": "confirmed_starting",
         "quote": quote, "source_url": "https://a"},
    ]
    assert pages == {"https://a": quote}


def test_run_agent_loop_recovers_from_invalid_json():
    responses = [
        "not json at all",
        json.dumps({"action": "final_answer", "classifications": []}),
    ]
    client = FakeAnthropicClient(responses)
    classifications, pages = run_agent_loop(
        client, "fake-model", "Leeds", fake_roster(), ToolBudget(),
    )
    assert classifications == []
    assert len(client.messages.calls) == 2


def test_run_agent_loop_search_budget_exceeded_is_surfaced_not_raised():
    responses = [
        json.dumps({"action": "search_web", "query": "q1"}),
        json.dumps({"action": "search_web", "query": "q2"}),
        json.dumps({"action": "final_answer", "classifications": []}),
    ]
    client = FakeAnthropicClient(responses)
    budget = ToolBudget(max_searches=1, max_fetches=1)
    search_calls = []

    def fake_search(query):
        search_calls.append(query)
        return []

    classifications, _pages = run_agent_loop(
        client, "fake-model", "Leeds", fake_roster(), budget,
        search_web=fake_search,
    )
    assert classifications == []
    assert search_calls == ["q1"]  # second search was refused, never called


def test_run_agent_loop_gives_up_after_max_turns_without_final_answer():
    responses = [json.dumps({"action": "search_web", "query": "q"})] * 3
    client = FakeAnthropicClient(responses)
    classifications, _pages = run_agent_loop(
        client, "fake-model", "Leeds", fake_roster(),
        ToolBudget(max_searches=10, max_fetches=10),
        search_web=lambda q: [], max_turns=3,
    )
    assert classifications == []


# --------------------------------------------------------------------------
# verify_classifications
# --------------------------------------------------------------------------


def test_verify_classifications_keeps_exact_substring_match():
    fetched = {"https://a": "Alice will start on Saturday."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "will start on Saturday", "source_url": "https://a"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


def test_verify_classifications_discards_quote_not_found_on_page():
    fetched = {"https://a": "Alice will start on Saturday."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "definitely benched", "source_url": "https://a"}],
        fetched, valid_codes={1001},
    )
    assert result == []


def test_verify_classifications_discards_code_not_on_roster():
    fetched = {"https://a": "Alice will start on Saturday."}
    result = verify_classifications(
        [{"code": 9999, "category": "confirmed_starting",
          "quote": "will start on Saturday", "source_url": "https://a"}],
        fetched, valid_codes={1001},
    )
    assert result == []


def test_verify_classifications_discards_unfetched_source_url():
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "anything", "source_url": "https://never-fetched"}],
        fetched_pages={}, valid_codes={1001},
    )
    assert result == []


# --------------------------------------------------------------------------
# predict_club_agent (end to end, all fakes)
# --------------------------------------------------------------------------


def test_predict_club_agent_applies_verified_classification_and_falls_back_others(tmp_path):
    rows = [{"element": i, "GW": g, "starts": 1} for i in (1, 2) for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001), (2, 1002)]),
    })
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[(1001, "Alice", 1), (1002, "Bob", 1)],
        teams_rows=[(1, "Leeds")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1002, SEASON, 1, 1)],
    )
    quote = "Alice will start on Saturday, the manager confirmed."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_starting",
             "quote": quote, "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)

    def fake_fetch_page(url):
        return quote

    frame = predict_club_agent(
        conn, SEASON, PRIOR_SEASON, 2, "Leeds",
        llm_client=client, fetch_page_text=fake_fetch_page, fetch=fetch,
    )
    conn.close()

    alice = frame[frame["code"] == 1001].iloc[0]
    bob = frame[frame["code"] == 1002].iloc[0]
    assert alice["method"] == "agent_confirmed_starting"
    assert alice["p_start"] == pytest.approx(0.90)  # unshrunk prior, no history yet
    assert alice["quote"] == quote
    assert bob["method"] == "agent_fallback_no_news"
    assert bob["category"] is None


def test_predict_club_agent_confirmed_out_hard_gates_to_zero(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[(1001, "Alice", 1)],
        teams_rows=[(1, "Leeds")],
        gameweek_rows=[(1001, SEASON, 1, 1)],
    )
    quote = "Alice is ruled out for Saturday with a hamstring injury."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out",
             "quote": quote, "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)

    frame = predict_club_agent(
        conn, SEASON, PRIOR_SEASON, 2, "Leeds",
        llm_client=client, fetch_page_text=lambda url: quote, fetch=fetch,
    )
    conn.close()

    row = frame[frame["code"] == 1001].iloc[0]
    assert row["p_start"] == 0.0
    assert row["method"] == "agent_confirmed_out"


def test_predict_club_agent_empty_roster_for_unknown_club(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[(1001, "Alice", 1)],
        teams_rows=[(1, "Leeds")],
        gameweek_rows=[(1001, SEASON, 1, 1)],
    )
    frame = predict_club_agent(
        conn, SEASON, PRIOR_SEASON, 2, "Nonexistent FC",
        llm_client=FakeAnthropicClient([]), fetch=fetch,
    )
    conn.close()
    assert len(frame) == 0
