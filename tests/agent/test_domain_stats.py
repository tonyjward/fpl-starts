"""Tests for agent/domain_stats.py: per-domain accuracy monitoring. No
network, no LLM -- rebuild_evidence_table reads plain on-disk JSON fixtures,
same as fpl_starts.derived's own tests do for predictions/ snapshots.
"""

import json
import os
import sqlite3

from fpl_starts.agent.domain_stats import (
    _domain_of,
    fit_domain_rates,
    rebuild_evidence_table,
)

SEASON = "2026-27"


def write_snapshot(predictions_dir, season, target_round, predicted_at, rows,
                    model_version="refined_availability_agent_news"):
    season_dir = os.path.join(predictions_dir, season)
    os.makedirs(season_dir, exist_ok=True)
    payload = {
        "season": season, "target_round": target_round,
        "model_version": model_version, "predicted_at": predicted_at,
        "predictions": rows,
    }
    name = "gw{0:02d}_{1}_{2}.json".format(target_round, model_version, predicted_at)
    with open(os.path.join(season_dir, name), "w") as f:
        json.dump(payload, f)


def make_evidence_row(code, category, quote, source_url):
    return {
        "code": code, "web_name": "Player{0}".format(code),
        "p_start": 0.5, "cold_start": False, "n_observed": 0,
        "method": "agent_" + category,
        "evidence": json.dumps([{"category": category, "quote": quote, "source_url": source_url}]),
    }


def make_derived_db(path, gameweek_rows):
    """gameweek_rows: (code, season, round, starts)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE player_gameweek_stats (code, season, round, starts)"
    )
    conn.executemany(
        "INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", gameweek_rows
    )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# _domain_of
# --------------------------------------------------------------------------


def test_domain_of_strips_www_and_path():
    assert _domain_of("https://www.SportsMole.co.uk/football/leeds?x=1") == "sportsmole.co.uk"


def test_domain_of_no_www_prefix_unchanged():
    assert _domain_of("https://bbc.co.uk/sport/football") == "bbc.co.uk"


def test_domain_of_empty_or_none_is_empty_string():
    assert _domain_of("") == ""
    assert _domain_of(None) == ""


# --------------------------------------------------------------------------
# rebuild_evidence_table
# --------------------------------------------------------------------------


def test_rebuild_evidence_table_populates_from_snapshot(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        make_evidence_row(1001, "confirmed_starting", "q1", "https://sportsmole.co.uk/a"),
        make_evidence_row(1002, "rotation_risk", "q2", "https://bbc.co.uk/b"),
    ])
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, predictions_dir, SEASON)
    assert n == 2
    rows = conn.execute(
        "SELECT code, category, domain FROM agent_evidence ORDER BY code"
    ).fetchall()
    assert rows == [
        (1001, "confirmed_starting", "sportsmole.co.uk"),
        (1002, "rotation_risk", "bbc.co.uk"),
    ]
    conn.close()


def test_rebuild_evidence_table_keeps_only_latest_snapshot_per_round(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        make_evidence_row(1001, "confirmed_starting", "old quote", "https://a"),
    ])
    write_snapshot(predictions_dir, SEASON, 3, "20260102T000000Z", [
        make_evidence_row(1001, "confirmed_out", "new quote", "https://a"),
    ])
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, predictions_dir, SEASON)
    assert n == 1
    row = conn.execute("SELECT category, quote FROM agent_evidence").fetchone()
    assert row == ("confirmed_out", "new quote")
    conn.close()


def test_rebuild_evidence_table_ignores_other_model_versions(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        {"code": 1001, "p_start": 0.8, "cold_start": False, "n_observed": 10,
         "method": "cal_rolling_xseason"},
    ], model_version="refined_availability")
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, predictions_dir, SEASON)
    assert n == 0
    conn.close()


def test_rebuild_evidence_table_skips_rows_with_no_evidence(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        {"code": 1001, "p_start": 0.5, "cold_start": False, "n_observed": 0,
         "method": "agent_fallback_no_news", "evidence": None},
    ])
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, predictions_dir, SEASON)
    assert n == 0
    conn.close()


def test_rebuild_evidence_table_missing_predictions_dir_is_a_noop(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, str(tmp_path / "nonexistent"), SEASON)
    assert n == 0
    conn.close()


def test_rebuild_evidence_table_multiple_evidence_items_expand_to_multiple_rows(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    row = {
        "code": 1001, "web_name": "Alice", "p_start": 0.6, "cold_start": False,
        "n_observed": 0, "method": "agent_blended",
        "evidence": json.dumps([
            {"category": "confirmed_starting", "quote": "q1", "source_url": "https://a"},
            {"category": "rotation_risk", "quote": "q2", "source_url": "https://b"},
        ]),
    }
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [row])
    conn = sqlite3.connect(str(tmp_path / "d.db"))
    n = rebuild_evidence_table(conn, predictions_dir, SEASON)
    assert n == 2
    conn.close()


# --------------------------------------------------------------------------
# fit_domain_rates
# --------------------------------------------------------------------------


def test_fit_domain_rates_counts_directional_hits_and_misses(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        make_evidence_row(1001, "confirmed_starting", "q1", "https://a.com/x"),
        make_evidence_row(1002, "confirmed_out", "q2", "https://a.com/y"),
    ])
    db_path = str(tmp_path / "d.db")
    conn = sqlite3.connect(db_path)
    rebuild_evidence_table(conn, predictions_dir, SEASON)
    conn.execute("CREATE TABLE player_gameweek_stats (code, season, round, starts)")
    conn.executemany("INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", [
        (1001, SEASON, 3, 1),  # confirmed_starting, actually started -> correct
        (1002, SEASON, 3, 1),  # confirmed_out, actually started -> wrong
    ])
    conn.commit()

    rates = fit_domain_rates(conn, SEASON, 4)
    conn.close()
    assert rates == {"a.com": (2, 1)}  # 2 directional claims, 1 correct


def test_fit_domain_rates_excludes_non_directional_categories(tmp_path):
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 3, "20260101T000000Z", [
        make_evidence_row(1001, "rotation_risk", "q1", "https://a.com/x"),
    ])
    db_path = str(tmp_path / "d.db")
    conn = sqlite3.connect(db_path)
    rebuild_evidence_table(conn, predictions_dir, SEASON)
    conn.execute("CREATE TABLE player_gameweek_stats (code, season, round, starts)")
    conn.executemany("INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", [
        (1001, SEASON, 3, 0),
    ])
    conn.commit()

    rates = fit_domain_rates(conn, SEASON, 4)
    conn.close()
    assert rates == {}


def test_fit_domain_rates_never_uses_the_round_being_scored(tmp_path):
    """Walk-forward: only rounds strictly before target_round count --
    scoring round 4 must not see round 4's own evidence.
    """
    predictions_dir = str(tmp_path / "predictions")
    write_snapshot(predictions_dir, SEASON, 4, "20260101T000000Z", [
        make_evidence_row(1001, "confirmed_starting", "q1", "https://a.com/x"),
    ])
    db_path = str(tmp_path / "d.db")
    conn = sqlite3.connect(db_path)
    rebuild_evidence_table(conn, predictions_dir, SEASON)
    conn.execute("CREATE TABLE player_gameweek_stats (code, season, round, starts)")
    conn.executemany("INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", [
        (1001, SEASON, 4, 1),
    ])
    conn.commit()

    rates = fit_domain_rates(conn, SEASON, 4)  # scoring round 4 itself
    conn.close()
    assert rates == {}


def test_fit_domain_rates_empty_with_no_evidence_at_all(tmp_path):
    conn = make_derived_db(str(tmp_path / "d.db"), gameweek_rows=[])
    conn.execute(
        "CREATE TABLE agent_evidence (code, season, target_round, category, "
        "quote, source_url, domain)"
    )
    conn.commit()
    assert fit_domain_rates(conn, SEASON, 4) == {}
    conn.close()
