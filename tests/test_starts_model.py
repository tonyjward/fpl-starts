"""Tests for starts_model.py.

The community archive is faked via an injected `fetch(path) -> bytes`
callable (never real network); derived.db is faked with a minimal sqlite
connection carrying just the columns starts_model.py actually reads, rather
than going through the full derived.rebuild() pipeline (already covered by
test_derived.py).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from fpl_starts import archiver
from fpl_starts import starts_model

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"


def make_fake_fetch(responses):
    def fetch(path):
        if path not in responses:
            raise AssertionError("unexpected path: {0}".format(path))
        return responses[path]
    return fetch


def merged_gw_csv(rows, gw_column="GW"):
    return pd.DataFrame(rows).rename(columns={"GW": gw_column}).to_csv(index=False).encode("utf-8")


def players_raw_csv(id_code_pairs):
    return pd.DataFrame(
        [{"id": i, "code": c} for i, c in id_code_pairs]
    ).to_csv(index=False).encode("utf-8")


def make_derived_db(path, players, gameweek_rows, availability_rows=None,
                     news_claims_rows=None, teams_rows=None):
    """A minimal sqlite db carrying only the columns starts_model.py
    reads from `players` / `player_gameweek_stats` /
    `player_availability_snapshots` / `news_claims` -- not a real
    derived.rebuild() output, which test_derived.py already covers.

    `availability_rows`: (code, season, fetched_at, next_gw, status, chance).
    `news_claims_rows`: (season, target_round, matched_code, category,
    contradiction, quote, content_type, source_tier, llm_p_start,
    claimed_opponent, opponent_mismatch). `quote` must be unique per row
    meant to count as a distinct claim -- dedupe_claims collapses
    same-quote rows within (code, category), same as production.
    `teams_rows`: (code, name) -- only needed by tests exercising the
    team-change reset (next_period_features's current_team cross-check);
    `players`/`player_gameweek_stats` carry `team_code` as an extra,
    inserted-by-name column so existing (code, web_name) / (code, season,
    round, starts) row tuples elsewhere in this file don't need to change.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE players (code INTEGER PRIMARY KEY, web_name TEXT, "
        "team_code INTEGER)"
    )
    conn.executemany(
        "INSERT INTO players (code, web_name) VALUES (?, ?)", players
    )
    conn.execute("CREATE TABLE teams (code INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO teams VALUES (?, ?)", teams_rows or [])
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
        "CREATE TABLE news_claims "
        "(season, target_round, matched_code, category, contradiction, "
        "quote, content_type, source_tier, llm_p_start, "
        "claimed_opponent, opponent_mismatch)"
    )
    conn.executemany(
        "INSERT INTO news_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        news_claims_rows or [],
    )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# load_prior_season_starts
# --------------------------------------------------------------------------


def test_load_prior_season_starts_joins_on_code_not_element():
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv([
            {"element": 1, "GW": 38, "starts": 1},
            {"element": 2, "GW": 38, "starts": 0},
        ]),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001), (2, 1002)]),
    })

    df = starts_model.load_prior_season_starts(fetch, PRIOR_SEASON)

    assert sorted(df["code"].tolist()) == [1001, 1002]
    assert int(df.loc[df["code"] == 1001, "y"].iloc[0]) == 1
    assert int(df.loc[df["code"] == 1002, "y"].iloc[0]) == 0


def test_load_prior_season_starts_falls_back_to_round_column():
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(
            [{"element": 1, "GW": 1, "starts": 1}], gw_column="round"
        ),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })

    df = starts_model.load_prior_season_starts(fetch, PRIOR_SEASON)

    assert df["GW"].iloc[0] == 1


# --------------------------------------------------------------------------
# load_current_season_starts
# --------------------------------------------------------------------------


def test_load_current_season_starts_reads_from_derived_db(tmp_path):
    conn = make_derived_db(str(tmp_path / "derived.db"), players=[(1001, "Alice")],
                            gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 0),
                                           (1001, PRIOR_SEASON, 1, 1)])

    df = starts_model.load_current_season_starts(conn, SEASON)
    conn.close()

    assert sorted(df["GW"].tolist()) == [1, 2]
    assert int(df.loc[df["GW"] == 2, "y"].iloc[0]) == 0


# --------------------------------------------------------------------------
# build_xseason_features -- continuity across the season boundary
# --------------------------------------------------------------------------


def test_features_are_continuous_across_season_boundary():
    """GW1 of the new season must see prior-season history in its prev/
    roll4, not NaN -- this is the entire point of the fix. Same club
    throughout, so team continuity is a non-issue here (see the
    team-change tests below for when it isn't).
    """
    prior_df = pd.DataFrame({
        "code": [1001] * 4,
        "GW": [35, 36, 37, 38],
        "y": [1, 1, 0, 1],
        "team": ["Arsenal"] * 4,
    })
    current_df = pd.DataFrame({"code": [1001], "GW": [1], "y": [1], "team": ["Arsenal"]})

    combined = starts_model.build_xseason_features(prior_df, current_df)

    gw1_row = combined[(combined["code"] == 1001) & (combined["period"] == 39)]
    assert gw1_row["prev"].iloc[0] == 1        # GW38's outcome
    assert gw1_row["roll4"].iloc[0] == 0.75    # mean of GW35-38 = [1,1,0,1]


# --------------------------------------------------------------------------
# build_xseason_features / next_period_features -- team-change reset
# --------------------------------------------------------------------------


def test_team_change_resets_prev_and_roll4():
    """A player who moved clubs must not carry the old club's form across
    the move -- their first game at the new club should see NaN prev/roll4
    (cold_start territory), not the old club's rolling average.
    """
    prior_df = pd.DataFrame({
        "code": [1001] * 4,
        "GW": [35, 36, 37, 38],
        "y": [1, 1, 1, 1],          # nailed on... at the old club
        "team": ["Everton"] * 4,
    })
    current_df = pd.DataFrame({
        "code": [1001, 1001],
        "GW": [1, 2],
        "y": [0, 1],
        "team": ["Man City", "Man City"],  # transferred
    })

    combined = starts_model.build_xseason_features(prior_df, current_df)

    first_at_new_club = combined[(combined["code"] == 1001) & (combined["period"] == 39)]
    assert first_at_new_club["prev"].isna().iloc[0]
    assert first_at_new_club["roll4"].isna().iloc[0]

    second_at_new_club = combined[(combined["code"] == 1001) & (combined["period"] == 40)]
    assert second_at_new_club["prev"].iloc[0] == 0     # GW1 at the new club, not GW38 at the old
    assert second_at_new_club["roll4"].iloc[0] == 0.0  # only GW1 at the new club counts


def test_team_change_within_current_season_also_resets():
    """The reset isn't only a season-boundary thing -- a mid-season
    transfer must break continuity too.
    """
    current_df = pd.DataFrame({
        "code": [1001] * 4,
        "GW": [1, 2, 3, 4],
        "y": [1, 1, 0, 1],
        "team": ["Brighton", "Brighton", "Newcastle", "Newcastle"],
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])

    combined = starts_model.build_xseason_features(prior_df, current_df)

    gw3 = combined[(combined["code"] == 1001) & (combined["period"] == 3)]
    assert gw3["prev"].isna().iloc[0]   # first game at Newcastle
    gw4 = combined[(combined["code"] == 1001) & (combined["period"] == 4)]
    assert gw4["prev"].iloc[0] == 0     # GW3 at Newcastle, not GW2 at Brighton


def test_missing_team_is_not_treated_as_a_break():
    """A missing `team` on either side isn't positive evidence of a
    transfer, so it must not trigger a reset -- same "ambiguous isn't
    evidence of a problem" stance derived.py's opponent_mismatch takes on
    an unresolvable claim. Only a *confirmed* differing team resets.
    """
    current_df = pd.DataFrame({
        "code": [1001, 1001],
        "GW": [1, 2],
        "y": [1, 1],
        "team": [None, "Arsenal"],
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])

    combined = starts_model.build_xseason_features(prior_df, current_df)

    gw2 = combined[(combined["code"] == 1001) & (combined["period"] == 2)]
    assert gw2["prev"].iloc[0] == 1


def test_next_period_features_only_uses_the_current_team_stint():
    """The live "next gameweek" features must reflect only the player's
    current club's history, even though build_xseason_features's row-level
    prev/roll4 (used for training) already does this per-row.
    """
    current_df = pd.DataFrame({
        "code": [1001] * 3,
        "GW": [1, 2, 3],
        "y": [1, 1, 0],
        "team": ["Everton", "Everton", "Man Utd"],   # transferred before GW3
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])

    combined = starts_model.build_xseason_features(prior_df, current_df)
    features = starts_model.next_period_features(combined)

    row = features[features["code"] == 1001].iloc[0]
    assert row["prev"] == 0        # GW3 at Man Utd, not GW2 (started) at Everton
    assert row["roll4"] == 0.0     # only the Man Utd appearance counts


def test_next_period_features_transfer_with_no_new_club_games_yet_is_excluded():
    """The case that matters most: a transfer announced *before* any
    gameweek has been played for the new club, so every archived row for
    this code is still at the old club and there's no team_stint break in
    the history to detect at all. Only a live current_team cross-check
    catches this -- code 1001 must be entirely absent from the result
    (which callers merging on `code` read as NaN/cold_start), even though
    every one of their historical rows looks perfectly continuous on its
    own.
    """
    current_df = pd.DataFrame({
        "code": [1001, 1001, 1002],
        "GW": [1, 2, 1],
        "y": [1, 1, 1],
        "team": ["Everton", "Everton", "Arsenal"],
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])
    combined = starts_model.build_xseason_features(prior_df, current_df)

    # 1001 has just transferred to Man Utd per live data; the archive
    # hasn't caught up yet. 1002 is unaffected.
    current_team = {1001: "Man Utd", 1002: "Arsenal"}
    features = starts_model.next_period_features(combined, current_team=current_team)

    assert 1001 not in set(features["code"])
    row_1002 = features[features["code"] == 1002].iloc[0]
    assert row_1002["prev"] == 1


def test_next_period_features_current_team_none_preserves_old_behaviour():
    """Omitting current_team (scoring.py's retrospective persistence
    baseline) must behave exactly as before this fix existed -- pure
    historical-stint detection, no live cross-check.
    """
    current_df = pd.DataFrame({
        "code": [1001, 1001],
        "GW": [1, 2],
        "y": [1, 1],
        "team": ["Everton", "Everton"],
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])
    combined = starts_model.build_xseason_features(prior_df, current_df)

    features = starts_model.next_period_features(combined)

    row = features[features["code"] == 1001].iloc[0]
    assert row["prev"] == 1


def test_next_period_features_current_team_unresolvable_is_not_guessed_stale():
    """A code current_team doesn't cover (e.g. team-name resolution
    failed) must be left alone, not assumed stale -- same "ambiguous isn't
    evidence of a problem" stance used elsewhere in this codebase.
    """
    current_df = pd.DataFrame({
        "code": [1001, 1001],
        "GW": [1, 2],
        "y": [1, 1],
        "team": ["Everton", "Everton"],
    })
    prior_df = pd.DataFrame(columns=["code", "GW", "y", "team"])
    combined = starts_model.build_xseason_features(prior_df, current_df)

    features = starts_model.next_period_features(combined, current_team={})

    row = features[features["code"] == 1001].iloc[0]
    assert row["prev"] == 1


# --------------------------------------------------------------------------
# fit_lookup_table / predict_p_start
# --------------------------------------------------------------------------


def test_fit_and_predict_recovers_observed_frequencies():
    train = pd.DataFrame({
        "prev": [1] * 60 + [0] * 60,
        "roll4": [1.0] * 60 + [0.0] * 60,
        "y": [1] * 54 + [0] * 6 + [0] * 57 + [1] * 3,
    })
    fitted = starts_model.fit_lookup_table(train)

    test = pd.DataFrame({"prev": [1, 0], "roll4": [1.0, 0.0]})
    predictions = starts_model.predict_p_start(fitted, test, min_cell=50)

    assert predictions[0] == pytest.approx(0.90)
    assert predictions[1] == pytest.approx(0.05)


def test_predict_falls_back_to_two_number_table_for_sparse_cells():
    train = pd.DataFrame({
        "prev": [1, 1, 1],
        "roll4": [0.5, 0.5, 1.0],   # the roll4=0.5 cell only has 2 rows
        "y": [1, 0, 1],
    })
    fitted = starts_model.fit_lookup_table(train)

    test = pd.DataFrame({"prev": [1], "roll4": [0.5]})
    predictions = starts_model.predict_p_start(fitted, test, min_cell=50)

    assert predictions[0] == fitted.r1  # fell back, not the 2-row cell's mean


# --------------------------------------------------------------------------
# predict_gameweek
# --------------------------------------------------------------------------


def test_predict_gameweek_round_one_uses_prior_season_rate(tmp_path):
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv([
            {"element": 1, "GW": g, "starts": 1} for g in range(1, 5)
        ] + [
            {"element": 1, "GW": g, "starts": 0} for g in range(5, 39)
        ]),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(str(tmp_path / "derived.db"),
                            players=[(1001, "Alice"), (9999, "NewSigning")],
                            gameweek_rows=[])

    result = starts_model.predict_gameweek(
        conn, SEASON, PRIOR_SEASON, target_round=1, fetch=fetch
    )
    conn.close()

    alice = result[result["code"] == 1001].iloc[0]
    assert alice["p_start"] == pytest.approx(4 / 38)
    assert alice["cold_start"] == False
    assert alice["method"] == "prior_season_rate"

    new_signing = result[result["code"] == 9999].iloc[0]
    assert new_signing["cold_start"] == True
    assert new_signing["method"] == "cold_start_pool_rate"


def test_predict_gameweek_later_round_uses_xseason_lookup_and_flags_cold_start(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice"), (9999, "NewSigning")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1)],
    )

    result = starts_model.predict_gameweek(
        conn, SEASON, PRIOR_SEASON, target_round=3, fetch=fetch
    )
    conn.close()

    alice = result[result["code"] == 1001].iloc[0]
    # Nailed-on every gameweek on both sides of the boundary -> high P(start).
    assert alice["p_start"] > 0.5
    assert alice["method"] == "cal_rolling_xseason"

    new_signing = result[result["code"] == 9999].iloc[0]
    assert new_signing["cold_start"] == True
    assert new_signing["n_observed"] == 0


# --------------------------------------------------------------------------
# load_availability_for_round
# --------------------------------------------------------------------------


def test_load_availability_for_round_keeps_latest_pull_per_player(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[],
        availability_rows=[
            (1001, SEASON, "20260903T090000Z", 3, "d", 50),
            (1001, SEASON, "20260903T180000Z", 3, "a", 100),
            (1001, SEASON, "20260902T090000Z", 2, "i", 0),  # wrong next_gw
        ],
    )

    result = starts_model.load_availability_for_round(conn, SEASON, 3)
    conn.close()

    row = result[result["code"] == 1001].iloc[0]
    assert row["status"] == "a"
    assert row["chance"] == 100


# --------------------------------------------------------------------------
# _flag_bucket
# --------------------------------------------------------------------------


def test_flag_bucket_by_chance_then_status():
    assert starts_model._flag_bucket("d", 75) == "chance_75"
    assert starts_model._flag_bucket("a", 25) == "chance_25"
    assert starts_model._flag_bucket("d", 0) == "chance_0"
    assert starts_model._flag_bucket("d", None) == "doubtful_no_chance"


# --------------------------------------------------------------------------
# route_predictions_with_availability
# --------------------------------------------------------------------------


def test_hard_gate_zeroes_unavailable_players():
    predictions = pd.DataFrame([
        {"code": 1001, "web_name": "Injured", "p_start": 0.85, "cold_start": False,
         "n_observed": 20, "method": "cal_rolling_xseason", "prev": 1},
        {"code": 1002, "web_name": "Fit", "p_start": 0.6, "cold_start": False,
         "n_observed": 20, "method": "cal_rolling_xseason", "prev": 1},
    ])
    availability = pd.DataFrame([
        {"code": 1001, "status": "i", "chance": 0},
        {"code": 1002, "status": "a", "chance": None},
    ])
    empty_flag_table = starts_model.FlagTable(
        cells=pd.DataFrame(columns=["mean", "size"]),
        pooled=pd.DataFrame(columns=["mean", "size"]),
    )

    result = starts_model.route_predictions_with_availability(
        predictions, availability, empty_flag_table
    )

    injured = result[result["code"] == 1001].iloc[0]
    assert injured["p_start"] == 0.0
    assert injured["method"] == "hard_gate_unavailable"

    fit = result[result["code"] == 1002].iloc[0]
    assert fit["p_start"] == 0.6  # untouched -- not injured/suspended/unavailable
    assert fit["method"] == "cal_rolling_xseason"


def test_flag_table_cell_used_when_populated_else_pooled_else_raw_kept():
    predictions = pd.DataFrame([
        {"code": 1001, "web_name": "WellPopulatedCell", "p_start": 0.7,
         "cold_start": False, "n_observed": 20, "method": "cal_rolling_xseason", "prev": 1},
        {"code": 1002, "web_name": "SparseCellPooledFallback", "p_start": 0.7,
         "cold_start": False, "n_observed": 20, "method": "cal_rolling_xseason", "prev": 1},
        {"code": 1003, "web_name": "NoDataAtAllKeepsRaw", "p_start": 0.7,
         "cold_start": False, "n_observed": 20, "method": "cal_rolling_xseason", "prev": 0},
    ])
    availability = pd.DataFrame([
        {"code": 1001, "status": "d", "chance": 75},
        {"code": 1002, "status": "d", "chance": 50},
        {"code": 1003, "status": "d", "chance": 25},
    ])
    flag_table = starts_model.FlagTable(
        cells=pd.DataFrame(
            {"mean": [0.3], "size": [60]},
            index=pd.MultiIndex.from_tuples([("chance_75", 1)], names=["bucket", "prev"]),
        ),
        pooled=pd.DataFrame({"mean": [0.4], "size": [60]}, index=pd.Index([1], name="prev")),
    )

    result = starts_model.route_predictions_with_availability(
        predictions, availability, flag_table, min_cell=50
    )

    assert result.loc[result["code"] == 1001, "p_start"].iloc[0] == 0.3
    assert result.loc[result["code"] == 1001, "method"].iloc[0] == "flag_table"

    assert result.loc[result["code"] == 1002, "p_start"].iloc[0] == 0.4
    assert result.loc[result["code"] == 1002, "method"].iloc[0] == "flag_table_pooled"

    # prev=0 has no pooled entry either -> falls all the way back to the
    # unmodified raw lookup prediction.
    assert result.loc[result["code"] == 1003, "p_start"].iloc[0] == 0.7
    assert result.loc[result["code"] == 1003, "method"].iloc[0] == "cal_rolling_xseason"


# --------------------------------------------------------------------------
# predict_gameweek_refined
# --------------------------------------------------------------------------


def test_predict_gameweek_refined_gates_injured_player(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1)],
        availability_rows=[(1001, SEASON, "20260903T090000Z", 3, "i", 0)],
    )

    raw = starts_model.predict_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    refined = starts_model.predict_gameweek_refined(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()

    assert raw.loc[raw["code"] == 1001, "p_start"].iloc[0] > 0.5  # nailed-on per history
    assert refined.loc[refined["code"] == 1001, "p_start"].iloc[0] == 0.0
    assert refined.loc[refined["code"] == 1001, "method"].iloc[0] == "hard_gate_unavailable"


def test_predict_gameweek_refined_falls_back_to_raw_with_no_availability_data(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1)],
        availability_rows=[],
    )

    raw = starts_model.predict_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    refined = starts_model.predict_gameweek_refined(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()

    pd.testing.assert_frame_equal(raw, refined)


# --------------------------------------------------------------------------
# snapshot_predictions
# --------------------------------------------------------------------------


def test_snapshot_predictions_write_once(tmp_path):
    predictions = pd.DataFrame([
        {"code": 1001, "web_name": "Alice", "p_start": 0.8, "cold_start": False,
         "n_observed": 10, "method": "cal_rolling_xseason"},
    ])
    base_dir = str(tmp_path / "predictions")
    clock = lambda: datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)

    path = starts_model.snapshot_predictions(
        predictions, SEASON, target_round=3, base_dir=base_dir, clock=clock
    )

    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["season"] == SEASON
    assert payload["target_round"] == 3
    assert payload["model_version"] == "raw_lookup"
    assert payload["predictions"][0]["web_name"] == "Alice"

    with pytest.raises(archiver.ArchiveError):
        starts_model.snapshot_predictions(
            predictions, SEASON, target_round=3, base_dir=base_dir, clock=clock
        )


def test_snapshot_predictions_different_model_versions_dont_collide(tmp_path):
    """Same round, same timestamp, different model_version -- both must be
    written, since they're different predictions worth comparing later.
    """
    predictions = pd.DataFrame([
        {"code": 1001, "web_name": "Alice", "p_start": 0.8, "cold_start": False,
         "n_observed": 10, "method": "cal_rolling_xseason"},
    ])
    base_dir = str(tmp_path / "predictions")
    clock = lambda: datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)

    raw_path = starts_model.snapshot_predictions(
        predictions, SEASON, target_round=3, model_version="raw_lookup",
        base_dir=base_dir, clock=clock,
    )
    refined_path = starts_model.snapshot_predictions(
        predictions, SEASON, target_round=3, model_version="refined_availability",
        base_dir=base_dir, clock=clock,
    )

    assert raw_path != refined_path
    assert os.path.exists(raw_path)
    assert os.path.exists(refined_path)


