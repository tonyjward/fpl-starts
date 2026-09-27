"""fpl_starts.refresh end to end on synthetic data: a real (synthetic) raw
archive, derived.db, frozen model and local data, with only the FPL API
call faked. Covers the deadline and cooldown gates, archiving, the atomic
rebuild, and registering a new forecast only when a chance changed."""

import json
import os
from datetime import timedelta, timezone

import pandas as pd
import pytest

from fpl_starts import archiver, derived, refresh
from fpl_starts.ml import logistic, spec

from ml.conftest import SyntheticWorld, deadline, training_frame
from test_derived import make_stats, write_ok_entry

SEASON = spec.PROSPECTIVE_SEASON
CODES = list(range(1000, 1008))
FILLER = 300  # bootstrap-static must list > 300 players to pass the archiver's checks
GW5_DEADLINE = deadline(SEASON, 5).tz_localize(timezone.utc).to_pydatetime()


def _bootstrap(statuses=None):
    """A plausible bootstrap-static before GW5: GW1-4 finished, 20 clubs,
    the 8 tracked players plus filler. `statuses`: {code: (status, chance, news)}."""
    statuses = statuses or {}
    events = [{"id": r, "deadline_time": deadline(SEASON, r).strftime("%Y-%m-%dT%H:%M:%SZ"), "finished": r <= 4,
               "data_checked": r <= 4, "is_current": r == 4, "is_next": r == 5} for r in range(1, 39)]
    teams = [{"id": i + 1, "code": 101 + i, "name": "Club{0}".format(i), "short_name": "C{0:02d}".format(i)}
             for i in range(20)]
    elements = []
    for i, code in enumerate(CODES):
        flag, chance, news = statuses.get(code, ("a", None, ""))
        elements.append({"id": i + 1, "code": code, "web_name": "Player{0}".format(i), "first_name": "First",
                         "second_name": "Player{0}".format(i), "team": 1 + i % 2, "element_type": 3,
                         "minutes": 360 if i < 3 else 0, "starts": 4 if i < 3 else 0, "status": flag,
                         "chance_of_playing_next_round": chance, "news": news, "now_cost": 50,
                         "selected_by_percent": "1.0"})
    for j in range(FILLER):
        elements.append({"id": 100 + j, "code": 5000 + j, "web_name": "Filler{0}".format(j), "team": 1 + j % 20,
                         "element_type": 2, "minutes": 0, "starts": 0, "status": "a",
                         "chance_of_playing_next_round": 100 if j == 0 else None, "news": "", "now_cost": 45,
                         "selected_by_percent": "0.1"})
    return {"events": events, "teams": teams, "elements": elements}


@pytest.fixture
def env(tmp_path):
    world = SyntheticWorld(str(tmp_path))
    world.fill_regular_season("2022-23", 6)
    for season in spec.TRAINING_SEASONS:
        world.fill_regular_season(season, 6)
    for r in (1, 2, 3, 4):
        for code in CODES:
            world.avail(SEASON, r, code)
    world.write()

    base_dir = str(tmp_path / "raw")
    for r in (1, 2, 3, 4):
        at = (deadline(SEASON, r) + pd.Timedelta(days=3)).tz_localize(timezone.utc).to_pydatetime()
        write_ok_entry(base_dir, SEASON, "event-live", r, at, {"elements": [
            {"id": i + 1, "stats": make_stats(90 if i < 3 else 0, 1 if i < 3 else 0)} for i in range(len(CODES))]})
    first = GW5_DEADLINE - timedelta(days=2)
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 4, first, _bootstrap(), next_gw=5,
                   next_deadline=deadline(SEASON, 5).strftime("%Y-%m-%dT%H:%M:%SZ"))

    paths = {"base_dir": base_dir, "db_path": str(tmp_path / "db" / "derived.db"),
             "predictions_dir": str(tmp_path / "predictions"), "models_dir": str(tmp_path / "models"),
             "data_dir": world.data_dir}
    derived.rebuild_and_swap(base_dir=base_dir, db_path=paths["db_path"], predictions_dir=paths["predictions_dir"])
    logistic.save(logistic.fit(training_frame(), 1.0, metadata={"created_at": "2026-08-01T00:00:00Z"}),
                  logistic.model_dir(paths["models_dir"]))
    return paths


class FakeFPL:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def __call__(self, path):
        assert path == "bootstrap-static/"
        self.calls += 1
        return 200, json.dumps(self.payload).encode()


def _refresh(env, fpl, at, **kwargs):
    return refresh.refresh(env["base_dir"], env["db_path"], env["predictions_dir"], env["models_dir"],
                           env["data_dir"], http_get=fpl, clock=lambda: at, **kwargs)


def _snapshots(env):
    d = os.path.join(env["predictions_dir"], SEASON)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _forecast(env, path):
    return {r["code"]: r for r in json.load(open(os.path.join(env["predictions_dir"], SEASON, path)))["predictions"]}


def test_refresh_archives_rebuilds_and_registers_a_new_forecast(env):
    fpl = FakeFPL(_bootstrap({1000: ("i", 0, "Knee injury - Expected back in 3 weeks")}))
    at = GW5_DEADLINE - timedelta(hours=1)  # after the old 2h cutoff, before the deadline
    result = _refresh(env, fpl, at)

    assert result.status == refresh.REFRESHED and result.gameweek == 5 and fpl.calls == 1
    assert result.forecast_updated and result.data_as_of == archiver.format_timestamp(at)
    assert _snapshots(env) == [os.path.basename(result.snapshot_path)]
    payload = json.load(open(result.snapshot_path))
    assert payload["prediction_cutoff"] == deadline(SEASON, 5).isoformat() + "Z"
    assert payload["generated_after_deadline"] is False
    forecast = _forecast(env, _snapshots(env)[0])
    assert forecast[1000]["availability_status"] == "injured"  # the news pulled an hour before the deadline
    assert forecast[1001]["availability_status"] == "available"

    import sqlite3
    conn = sqlite3.connect(env["db_path"])  # the new capture and the new forecast are both in derived.db
    assert conn.execute("SELECT news FROM player_availability_snapshots WHERE code = 1000 AND fetched_at = ?",
                        (result.data_as_of,)).fetchone()[0].startswith("Knee injury")
    assert conn.execute("SELECT COUNT(*) FROM predictions WHERE target_round = 5").fetchone()[0] == len(forecast)
    archiver.verify_archive(env["base_dir"], SEASON)


def test_a_recent_refresh_is_not_repeated(env):
    fpl = FakeFPL(_bootstrap())
    first = GW5_DEADLINE - timedelta(days=1)
    assert _refresh(env, fpl, first).status == refresh.REFRESHED
    again = _refresh(env, fpl, first + timedelta(minutes=10))
    assert again.status == refresh.RECENT and fpl.calls == 1
    assert again.data_as_of == archiver.format_timestamp(first)


def test_unchanged_data_registers_no_new_forecast(env):
    fpl = FakeFPL(_bootstrap())
    first = GW5_DEADLINE - timedelta(days=1)
    assert _refresh(env, fpl, first).forecast_updated  # the first forecast for GW5
    second = _refresh(env, fpl, first + timedelta(hours=1))
    assert second.status == refresh.REFRESHED and fpl.calls == 2
    assert not second.forecast_updated and "no player's chance of starting changed" in second.message
    assert len(_snapshots(env)) == 1


def test_no_refresh_after_the_deadline(env):
    fpl = FakeFPL(_bootstrap())
    for at in (GW5_DEADLINE, GW5_DEADLINE + timedelta(days=1)):
        result = _refresh(env, fpl, at)
        assert result.status == refresh.DEADLINE_PASSED and "gameweek 5 deadline" in result.message
    assert fpl.calls == 0 and _snapshots(env) == []


def test_the_cooldown_is_configurable(env):
    fpl = FakeFPL(_bootstrap())
    first = GW5_DEADLINE - timedelta(days=1)
    _refresh(env, fpl, first)
    assert _refresh(env, fpl, first + timedelta(minutes=10), cooldown=timedelta(minutes=5)).status == refresh.REFRESHED
    assert fpl.calls == 2
