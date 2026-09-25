"""Availability must be strictly pre-cutoff, and missing local inputs must
fail loudly rather than silently dropping availability."""

import os

import pandas as pd
import pytest

from fpl_starts.ml import data as mldata
from fpl_starts.ml import panel as mlpanel
from fpl_starts.ml import spec

from .conftest import add_snapshot, deadline

SEASON = spec.PROSPECTIVE_SEASON


def _cutoffs(world):
    return mlpanel.current_season_cutoffs(world.bootstrap(0), SEASON)


def test_check_pre_cutoff_rejects_a_snapshot_at_or_after_the_cutoff():
    cutoff = pd.Timestamp("2025-08-15 15:00")
    ok = pd.DataFrame({"season": ["2025-26"], "round": [1], "snapshot_at": [cutoff - pd.Timedelta(seconds=1)]})
    mldata.check_pre_cutoff(ok, {("2025-26", 1): cutoff})
    for late in (cutoff, cutoff + pd.Timedelta(hours=1)):
        bad = ok.assign(snapshot_at=[late])
        with pytest.raises(ValueError, match="strictly before"):
            mldata.check_pre_cutoff(bad, {("2025-26", 1): cutoff})


def test_check_pre_cutoff_rejects_rows_with_no_known_cutoff():
    rows = pd.DataFrame({"season": ["2025-26"], "round": [5], "snapshot_at": [pd.Timestamp("2025-01-01")]})
    with pytest.raises(ValueError, match="no prediction cutoff"):
        mldata.check_pre_cutoff(rows, {})


def test_archived_snapshots_after_the_cutoff_are_ignored_and_latest_prior_wins(world):
    world.write()
    conn = world.db()
    cutoff = deadline(SEASON, 5) - pd.Timedelta(hours=spec.CUTOFF_HOURS_BEFORE_DEADLINE)
    add_snapshot(conn, 1, 5, cutoff - pd.Timedelta(days=2), status="a")
    add_snapshot(conn, 1, 5, cutoff - pd.Timedelta(hours=1), status="d", chance=50)  # latest pre-cutoff
    add_snapshot(conn, 1, 5, cutoff + pd.Timedelta(minutes=1), status="i", chance=0)  # after cutoff
    add_snapshot(conn, 2, 5, cutoff + pd.Timedelta(minutes=1), status="i", chance=0)  # only post-cutoff
    add_snapshot(conn, 3, 5, cutoff - pd.Timedelta(hours=5))
    got = mlpanel.load_current_season_availability(conn, world.data_dir, SEASON, [5], _cutoffs(world))
    assert set(got["code"]) == {1, 3}
    assert got.set_index("code").loc[1, "status"] == "d"
    assert (got["snapshot_at"] < cutoff).all()


def test_a_round_with_no_pre_cutoff_availability_raises(world):
    world.write()
    conn = world.db()
    cutoff = deadline(SEASON, 6) - pd.Timedelta(hours=spec.CUTOFF_HOURS_BEFORE_DEADLINE)
    add_snapshot(conn, 1, 6, cutoff + pd.Timedelta(hours=1))
    with pytest.raises(mldata.MissingLocalDataError, match="no pre-cutoff availability"):
        mlpanel.load_current_season_availability(conn, world.data_dir, SEASON, [6], _cutoffs(world))


def test_local_availability_file_rows_after_the_cutoff_raise(world):
    world.avail(SEASON, 3, 1, hours_before_cutoff=-1.0)  # taken an hour after the cutoff
    world.write()
    with pytest.raises(ValueError, match="strictly before"):
        mlpanel.load_current_season_availability(world.db(), world.data_dir, SEASON, [3], _cutoffs(world))


def test_historical_panel_rejects_post_cutoff_availability(world):
    world.fill_regular_season("2022-23", 3)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 3)
    world.avail("2024-25", 2, 1000, hours_before_cutoff=-0.5)
    world.write()
    with pytest.raises(ValueError, match="strictly before"):
        mlpanel.build_historical_panel(world.data_dir)


@pytest.mark.parametrize("relative", [
    os.path.join("availability", "historical_availability.csv"),
    os.path.join("deadlines", "gameweek_deadlines.csv"),
    os.path.join("vaastav", "2024-25", "gws", "merged_gw.csv"),
])
def test_missing_local_data_fails_loudly(world, relative):
    world.fill_regular_season("2022-23", 3)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 3)
    world.write()
    os.remove(os.path.join(world.data_dir, relative))
    with pytest.raises(mldata.MissingLocalDataError, match="not distributed"):
        mlpanel.build_historical_panel(world.data_dir)


def test_historical_rows_without_a_snapshot_are_unknown_not_dropped(world):
    world.fill_regular_season("2022-23", 3)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 3)
    world.availability = [a for a in world.availability
                          if not (a["season"] == "2024-25" and a["gameweek"] == 2 and a["code"] == 1001)]
    world.write()
    panel = mlpanel.build_historical_panel(world.data_dir)
    row = panel[(panel["season"] == "2024-25") & (panel["round"] == 2) & (panel["code"] == 1001)]
    assert len(row) == 1 and row.iloc[0]["availability_status"] == "unknown"
