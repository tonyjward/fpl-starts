"""Tests for agent/categories.py: the classify-to-probability lookup.
No LLM or network involved anywhere in this file -- this is pure arithmetic
plus one sqlite query, same as starts_model.py's own shrink-adjacent tests.
"""

import sqlite3

from fpl_starts.agent.categories import (
    CATEGORY_PRIORS,
    category_to_p_start,
    fit_category_rates,
    shrink,
)

SEASON = "2026-27"


def make_agent_db(path, prediction_rows, gameweek_rows):
    """prediction_rows: (code, season, target_round, model_version, method).
    gameweek_rows: (code, season, round, starts).
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE predictions (code, season, target_round, "
        "model_version, method)"
    )
    conn.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", prediction_rows
    )
    conn.execute(
        "CREATE TABLE player_gameweek_stats (code, season, round, starts)"
    )
    conn.executemany(
        "INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", gameweek_rows
    )
    conn.commit()
    return conn


def test_shrink_with_no_observations_is_exactly_the_prior():
    assert shrink(0.90, 0.0, 0, k=10) == 0.90


def test_shrink_converges_toward_observed_rate_as_n_grows():
    near_prior = shrink(0.90, 5, 10, k=10)
    more_data = shrink(0.90, 50, 100, k=10)
    # observed rate here is 0.5, well below the 0.90 prior -- more
    # observations should pull the estimate further from the prior.
    assert abs(more_data - 0.5) < abs(near_prior - 0.5)


def test_category_to_p_start_confirmed_out_is_always_zero_regardless_of_rates():
    assert category_to_p_start("confirmed_out", {"confirmed_out": (100, 100)}) == 0.0


def test_category_to_p_start_unknown_category_returns_none():
    assert category_to_p_start("mystery", {}) is None


def test_category_to_p_start_uses_prior_when_no_observed_rate_yet():
    p = category_to_p_start("rotation_risk", {})
    assert p == CATEGORY_PRIORS["rotation_risk"]


def test_fit_category_rates_empty_with_no_prior_rounds(tmp_path):
    conn = make_agent_db(str(tmp_path / "d.db"), [], [])
    assert fit_category_rates(conn, SEASON, 4) == {}
    conn.close()


def test_fit_category_rates_only_uses_rounds_strictly_before_target(tmp_path):
    conn = make_agent_db(
        str(tmp_path / "d.db"),
        prediction_rows=[
            (1, SEASON, 3, "refined_availability_agent_news", "agent_rotation_risk"),
            (2, SEASON, 4, "refined_availability_agent_news", "agent_rotation_risk"),
        ],
        gameweek_rows=[(1, SEASON, 3, 0), (2, SEASON, 4, 1)],
    )
    rates = fit_category_rates(conn, SEASON, 4)
    assert rates == {"rotation_risk": (0.0, 1)}
    conn.close()


def test_fit_category_rates_excludes_fallback_rows(tmp_path):
    conn = make_agent_db(
        str(tmp_path / "d.db"),
        prediction_rows=[
            (1, SEASON, 3, "refined_availability_agent_news", "agent_fallback_no_news"),
            (2, SEASON, 3, "refined_availability_agent_news", "agent_confirmed_starting"),
        ],
        gameweek_rows=[(1, SEASON, 3, 1), (2, SEASON, 3, 1)],
    )
    rates = fit_category_rates(conn, SEASON, 4)
    assert rates == {"confirmed_starting": (1.0, 1)}
    conn.close()


def test_fit_category_rates_ignores_other_model_versions(tmp_path):
    conn = make_agent_db(
        str(tmp_path / "d.db"),
        prediction_rows=[
            (1, SEASON, 3, "refined_availability", "flag_table"),
        ],
        gameweek_rows=[(1, SEASON, 3, 1)],
    )
    rates = fit_category_rates(conn, SEASON, 4)
    assert rates == {}
    conn.close()
