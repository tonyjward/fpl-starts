"""Tests for agent/predict.py: the manual tool-use loop, quote verification,
and the classify-then-lookup end-to-end flow. No real network or Anthropic
API calls anywhere -- llm_client/search_web/fetch_page_text are all fakes.
"""

import json
import sqlite3

import pandas as pd
import pytest

from fpl_starts.agent.categories import CATEGORY_PRIORS
from fpl_starts.agent.predict import (
    _blend_verified,
    get_opponent,
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


def make_derived_db(path, players, teams_rows, gameweek_rows, fixtures_rows=None,
                     availability_rows=None):
    """players: (code, web_name, team_code). teams_rows: (code, name).
    fixtures_rows: (season, round, team_code, opponent_code) -- one row per
    side, same as fpl_starts.derived's real _load_fixtures output.
    availability_rows: (code, season, fetched_at, next_gw, status, chance).
    """
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
    conn.executemany(
        "INSERT INTO player_availability_snapshots VALUES (?, ?, ?, ?, ?, ?)",
        availability_rows or [],
    )
    conn.execute(
        "CREATE TABLE predictions (code, season, target_round, model_version, method)"
    )
    conn.execute(
        "CREATE TABLE fixtures (season, round, team_code, opponent_code)"
    )
    conn.executemany(
        "INSERT INTO fixtures VALUES (?, ?, ?, ?)", fixtures_rows or []
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
        # Snapshot messages at call time -- predict.py mutates the same
        # list object turn to turn, so storing a live reference here would
        # make an earlier call's recorded messages silently grow to include
        # messages appended after it, in real HTTP behavior it's serialized
        # at call time.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
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


def test_get_opponent_resolves_from_fixtures_table(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[], teams_rows=[(1, "Leeds"), (2, "Newcastle")],
        gameweek_rows=[],
        fixtures_rows=[(SEASON, 4, 1, 2), (SEASON, 4, 2, 1)],
    )
    assert get_opponent(conn, SEASON, 4, "Leeds") == "Newcastle"
    assert get_opponent(conn, SEASON, 4, "Newcastle") == "Leeds"
    conn.close()


def test_get_opponent_none_when_no_fixture_recorded(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[], teams_rows=[(1, "Leeds")], gameweek_rows=[],
    )
    assert get_opponent(conn, SEASON, 4, "Leeds") is None
    conn.close()


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


def test_run_agent_loop_includes_opponent_in_initial_prompt_when_known():
    client = FakeAnthropicClient([
        json.dumps({"action": "final_answer", "classifications": []}),
    ])
    run_agent_loop(client, "fake-model", "Leeds", fake_roster(), ToolBudget(),
                    opponent_name="Newcastle")
    first_call_messages = client.messages.calls[0]["messages"]
    assert "Newcastle" in first_call_messages[0]["content"]


def test_run_agent_loop_flags_unknown_fixture_when_opponent_not_given():
    client = FakeAnthropicClient([
        json.dumps({"action": "final_answer", "classifications": []}),
    ])
    run_agent_loop(client, "fake-model", "Leeds", fake_roster(), ToolBudget())
    first_call_messages = client.messages.calls[0]["messages"]
    assert "unknown" in first_call_messages[0]["content"]


def test_run_agent_loop_gives_the_model_a_corrective_turn_for_a_hedged_confirmed_out():
    bad_quote = "Doubtful: Amar Dedic (hamstring), Ewen Jaouen (ankle)"
    good_quote = "Doubtful: Amar Dedic (hamstring), Ewen Jaouen (ankle)"
    responses = [
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out",
             "quote": bad_quote, "source_url": "https://a"},
        ]}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "rotation_risk",
             "quote": good_quote, "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)

    classifications, _pages = run_agent_loop(
        client, "fake-model", "Newcastle", fake_roster(), ToolBudget(),
    )

    assert len(client.messages.calls) == 2  # the corrective turn actually happened
    assert classifications == [
        {"code": 1001, "category": "rotation_risk",
         "quote": good_quote, "source_url": "https://a"},
    ]
    # the corrective message named the specific problem
    second_call_messages = client.messages.calls[1]["messages"]
    corrective = second_call_messages[-1]["content"]
    assert "1001" in corrective and "hedged" in corrective


def test_run_agent_loop_returns_still_flagged_answer_after_reclassification_budget_used():
    bad_quote = "Doubtful: Ewen Jaouen (ankle)"
    responses = [
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out", "quote": bad_quote,
             "source_url": "https://a"},
        ]}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out", "quote": bad_quote,
             "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)

    classifications, _pages = run_agent_loop(
        client, "fake-model", "Newcastle", fake_roster(), ToolBudget(),
        max_reclassifications=1,
    )

    # one corrective turn used, model didn't fix it, loop doesn't keep going
    assert len(client.messages.calls) == 2
    assert classifications[0]["category"] == "confirmed_out"  # unfixed, returned as-is


def test_run_agent_loop_accepts_non_hedged_confirmed_out_without_a_corrective_turn():
    responses = [
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out",
             "quote": "Out: Joe Rodon (thigh)", "source_url": "https://a"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)

    classifications, _pages = run_agent_loop(
        client, "fake-model", "Leeds", fake_roster(), ToolBudget(),
    )
    assert len(client.messages.calls) == 1  # no corrective turn needed
    assert classifications[0]["category"] == "confirmed_out"


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


def test_verify_classifications_discards_match_report_content_type():
    fetched = {"https://a": "Alice was sent off in yesterday's match."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_out", "content_type": "match_report",
          "quote": "was sent off in yesterday's match", "source_url": "https://a"}],
        fetched, valid_codes={1001},
    )
    assert result == []


def test_verify_classifications_keeps_team_news_content_type():
    fetched = {"https": "Alice will start on Saturday."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting", "content_type": "team_news",
          "quote": "will start on Saturday", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


def test_verify_classifications_missing_content_type_defaults_to_kept():
    """No content_type field at all (e.g. an older/partial response) should
    not itself be a reason to discard -- only an explicit match_report is.
    """
    fetched = {"https": "Alice will start on Saturday."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "will start on Saturday", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


def test_verify_classifications_discards_wrong_opponent():
    fetched = {"https": "Alice will miss the trip to Crystal Palace."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_out", "opponent": "Crystal Palace",
          "quote": "will miss the trip to Crystal Palace", "source_url": "https"}],
        fetched, valid_codes={1001}, opponent_name="Newcastle",
    )
    assert result == []


def test_verify_classifications_keeps_matching_opponent_case_insensitive():
    fetched = {"https": "Alice will start against newcastle."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting", "opponent": "NEWCASTLE",
          "quote": "will start against newcastle", "source_url": "https"}],
        fetched, valid_codes={1001}, opponent_name="Newcastle",
    )
    assert len(result) == 1


def test_verify_classifications_empty_opponent_is_never_a_mismatch():
    fetched = {"https": "Alice will start this weekend."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting", "opponent": "",
          "quote": "will start this weekend", "source_url": "https"}],
        fetched, valid_codes={1001}, opponent_name="Newcastle",
    )
    assert len(result) == 1


def test_verify_classifications_no_known_opponent_skips_the_check():
    """opponent_name=None (no fixture on record) means the opponent check
    is skipped entirely -- a claimed opponent is neither confirmed nor
    contradicted when there's nothing to check it against.
    """
    fetched = {"https": "Alice will miss the trip to Crystal Palace."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_out", "opponent": "Crystal Palace",
          "quote": "will miss the trip to Crystal Palace", "source_url": "https"}],
        fetched, valid_codes={1001}, opponent_name=None,
    )
    assert len(result) == 1


def test_verify_classifications_matches_curly_quotes_against_straight():
    fetched = {"https": "Alice ‘will start’ on Saturday, he said."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "'will start' on Saturday", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


def test_verify_classifications_discards_hedged_confirmed_out_as_final_safety_net():
    """A classification that slipped past run_agent_loop's own correction
    turn unflagged (e.g. verify_classifications called directly, as
    predict_club_agent does) should still not be trusted at face value.
    """
    fetched = {"https": "Doubtful: Ewen Jaouen (ankle)."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_out",
          "quote": "Doubtful: Ewen Jaouen (ankle)", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert result == []


def test_verify_classifications_keeps_hedged_rotation_risk():
    """Hedging language is exactly what rotation_risk/returning_from_injury
    are for -- only confirmed_out/confirmed_starting are held to this
    standard.
    """
    fetched = {"https": "Doubtful: Ewen Jaouen (ankle)."}
    result = verify_classifications(
        [{"code": 1001, "category": "rotation_risk",
          "quote": "Doubtful: Ewen Jaouen (ankle)", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


def test_verify_classifications_matches_regardless_of_case_and_whitespace():
    fetched = {"https": "Alice will\n  start   on Saturday."}
    result = verify_classifications(
        [{"code": 1001, "category": "confirmed_starting",
          "quote": "WILL START ON saturday", "source_url": "https"}],
        fetched, valid_codes={1001},
    )
    assert len(result) == 1


# --------------------------------------------------------------------------
# _blend_verified
# --------------------------------------------------------------------------


def test_blend_verified_single_item_prices_at_the_category_prior():
    result = _blend_verified(
        [{"category": "rotation_risk", "quote": "q1"}], category_rates={},
    )
    assert result == (CATEGORY_PRIORS["rotation_risk"], "agent_rotation_risk", False)


def test_blend_verified_unanimous_confirmed_out_forces_zero():
    result = _blend_verified(
        [{"category": "confirmed_out", "quote": "q1"},
         {"category": "confirmed_out", "quote": "q2"}],
        category_rates={},
    )
    assert result == (0.0, "agent_confirmed_out", True)


def test_blend_verified_two_disagreeing_categories_is_the_mean_and_not_forced():
    result = _blend_verified(
        [{"category": "confirmed_starting", "quote": "q1"},
         {"category": "rotation_risk", "quote": "q2"}],
        category_rates={},
    )
    p_start, method, forced = result
    expected = (CATEGORY_PRIORS["confirmed_starting"] + CATEGORY_PRIORS["rotation_risk"]) / 2
    assert p_start == pytest.approx(expected)
    assert method == "agent_blended"
    assert forced is False


def test_blend_verified_mixed_set_including_confirmed_out_is_not_forced():
    """A *mixed* set containing confirmed_out alongside another category is
    not unanimous -- it goes through the normal weighted blend (where
    confirmed_out prices at 0.0, pulling the average down) rather than
    being treated as a certain hard gate.
    """
    result = _blend_verified(
        [{"category": "confirmed_out", "quote": "q1"},
         {"category": "confirmed_starting", "quote": "q2"}],
        category_rates={},
    )
    p_start, method, forced = result
    expected = (0.0 + CATEGORY_PRIORS["confirmed_starting"]) / 2
    assert p_start == pytest.approx(expected)
    assert method == "agent_blended"
    assert forced is False


def test_blend_verified_duplicate_quotes_count_once_not_twice():
    """Same quote (post-normalization) reported twice must not get double
    weight in the blend -- same reasoning as the private repo's own
    dedupe_claims fix for syndicated content.
    """
    result = _blend_verified(
        [{"category": "confirmed_starting", "quote": "He will start on Saturday."},
         {"category": "rotation_risk", "quote": "he WILL start on saturday."}],
        category_rates={},
    )
    # both items normalize to the same quote -> only the first survives,
    # so this is a single confirmed_starting classification, not a blend.
    assert result == (CATEGORY_PRIORS["confirmed_starting"], "agent_confirmed_starting", False)


def test_blend_verified_unknown_category_is_skipped_not_fatal():
    result = _blend_verified(
        [{"category": "mystery_category", "quote": "q1"},
         {"category": "rotation_risk", "quote": "q2"}],
        category_rates={},
    )
    assert result == (CATEGORY_PRIORS["rotation_risk"], "agent_rotation_risk", False)


def test_blend_verified_returns_none_when_nothing_priceable():
    result = _blend_verified(
        [{"category": "mystery_category", "quote": "q1"}], category_rates={},
    )
    assert result is None


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
    assert json.loads(alice["evidence"]) == [
        {"category": "confirmed_starting", "quote": quote, "source_url": "https://a"},
    ]
    assert bob["method"] == "agent_fallback_no_news"
    assert bob["evidence"] is None


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


def test_predict_club_agent_defers_to_an_existing_fpl_hard_gate(tmp_path):
    """A rotation_risk (or any non-confirmed_out) classification must not
    override a player FPL's own status already hard-gated to 0 -- same
    precedence the private repo's route_predictions_with_news uses.
    """
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
        # status "i" (injured) at next_gw=2 -> predict_gameweek_refined
        # hard-gates Alice to 0.0 for target_round=2 before the agent
        # even runs.
        availability_rows=[(1001, SEASON, "20260101T000000Z", 2, "i", None)],
    )
    quote = "Alice could be rotated for Saturday's game."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "rotation_risk",
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
    assert row["p_start"] == 0.0  # unchanged -- the anchor's own hard gate
    assert row["method"] == "agent_deferred_to_availability"
    assert json.loads(row["evidence"]) == [
        {"category": "rotation_risk", "quote": quote, "source_url": "https://a"},
    ]  # still recorded, just not applied


def test_predict_club_agent_confirmed_out_still_applies_over_an_existing_hard_gate(tmp_path):
    """confirmed_out is the deliberate exception to the precedence rule --
    it applies regardless, same as the private repo's ordering (the
    confirmed_out check happens before the decided-status check).
    """
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
        availability_rows=[(1001, SEASON, "20260101T000000Z", 2, "i", None)],
    )
    quote = "Alice is definitely out for Saturday."
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
    assert row["method"] == "agent_confirmed_out"  # applied, not deferred


def test_predict_club_agent_blends_two_disagreeing_sources_for_one_player(tmp_path):
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
    quote_a = "Alice will start on Saturday, the manager confirmed."
    quote_b = "Alice is a doubt for the weekend with a knock."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "fetch_page_text", "url": "https://b"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_starting",
             "quote": quote_a, "source_url": "https://a"},
            {"code": 1001, "category": "rotation_risk",
             "quote": quote_b, "source_url": "https://b"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)
    pages = {"https://a": quote_a, "https://b": quote_b}

    frame = predict_club_agent(
        conn, SEASON, PRIOR_SEASON, 2, "Leeds",
        llm_client=client, fetch_page_text=lambda url: pages[url], fetch=fetch,
    )
    conn.close()

    row = frame[frame["code"] == 1001].iloc[0]
    expected = (CATEGORY_PRIORS["confirmed_starting"] + CATEGORY_PRIORS["rotation_risk"]) / 2
    assert row["p_start"] == pytest.approx(expected)
    assert row["method"] == "agent_blended"
    assert len(json.loads(row["evidence"])) == 2


def test_predict_club_agent_mixed_confirmed_out_not_unanimous_still_defers(tmp_path):
    """A mixed set (confirmed_out alongside another category) is not
    unanimous, so it does NOT get the forced-apply exception -- it's
    subject to the same availability-precedence check as anything else.
    """
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
        availability_rows=[(1001, SEASON, "20260101T000000Z", 2, "i", None)],
    )
    quote_a = "Alice is ruled out for Saturday."
    quote_b = "Alice will start on Saturday, the manager confirmed."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "fetch_page_text", "url": "https://b"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "confirmed_out",
             "quote": quote_a, "source_url": "https://a"},
            {"code": 1001, "category": "confirmed_starting",
             "quote": quote_b, "source_url": "https://b"},
        ]}),
    ]
    client = FakeAnthropicClient(responses)
    pages = {"https://a": quote_a, "https://b": quote_b}

    frame = predict_club_agent(
        conn, SEASON, PRIOR_SEASON, 2, "Leeds",
        llm_client=client, fetch_page_text=lambda url: pages[url], fetch=fetch,
    )
    conn.close()

    row = frame[frame["code"] == 1001].iloc[0]
    assert row["p_start"] == 0.0  # unchanged -- the anchor's own hard gate wins
    assert row["method"] == "agent_deferred_to_availability"
    assert len(json.loads(row["evidence"])) == 2  # both still recorded


def test_predict_club_agent_applies_normally_when_anchor_has_no_availability_decision(tmp_path):
    """The precedence check only ever blocks the three
    _AVAILABILITY_DECIDED_METHODS values -- an ordinary cal_rolling_xseason
    anchor (no FPL-status decision at all) is fair game for the agent.
    """
    rows = [{"element": i, "GW": g, "starts": 1} for i in (1, 2) for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001), (2, 1002)]),
    })
    conn = make_derived_db(
        str(tmp_path / "d.db"),
        players=[(1001, "Alice", 1)],
        teams_rows=[(1, "Leeds")],
        gameweek_rows=[(1001, SEASON, 1, 1)],
        # No availability_rows at all -- predict_gameweek_refined falls
        # back to the plain lookup table, method="cal_rolling_xseason".
    )
    quote = "Alice could be rotated for Saturday's game."
    responses = [
        json.dumps({"action": "fetch_page_text", "url": "https://a"}),
        json.dumps({"action": "final_answer", "classifications": [
            {"code": 1001, "category": "rotation_risk",
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
    assert row["method"] == "agent_rotation_risk"  # applied normally


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
