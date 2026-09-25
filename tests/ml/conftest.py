"""Synthetic local inputs for the logistic P(start) tests -- never real data.

`SyntheticWorld` writes a tiny data/ tree (vaastav-format seasons, availability
files, deadlines) and a derived.db, so the real loaders and feature code run
end to end on a handful of invented players and teams.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fpl_starts import derived
from fpl_starts.ml import spec

AVAILABILITY_COLUMNS = ["season", "gameweek", "snapshot_timestamp", "snapshot_age_hours", "code", "id",
                        "status", "chance_of_playing_next_round", "chance_of_playing_this_round", "news",
                        "news_added"]
TEAMS = [(1, 101, "Alpha"), (2, 102, "Bravo"), (3, 103, "Charlie"), (4, 104, "Delta")]
SEASON_START = {"2022-23": "2022-08-05", "2023-24": "2023-08-11", "2024-25": "2024-08-16",
                "2025-26": "2025-08-15", "2026-27": "2026-08-21"}


def deadline(season, rnd):
    return pd.Timestamp(SEASON_START[season]) + pd.Timedelta(days=7 * (rnd - 1), hours=17)


class SyntheticWorld:
    def __init__(self, root):
        self.data_dir = os.path.join(root, "data")
        self.db_path = os.path.join(root, "derived.db")
        self.gw_rows = {}       # season -> list of merged_gw dicts
        self.players = {}       # season -> {element id: (code, element_type)}
        self.availability = []  # rows of the availability-file schema

    # -- historical seasons ------------------------------------------------
    def player(self, season, element, code, element_type=3):
        self.players.setdefault(season, {})[element] = (code, element_type)

    def appearance(self, season, element, rnd, team_name, minutes, started, fixture=None):
        self.gw_rows.setdefault(season, []).append({
            "element": element, "GW": rnd, "fixture": fixture if fixture is not None else rnd * 10,
            "team": team_name, "minutes": minutes, "starts": int(started),
            "kickoff_time": str(deadline(season, rnd) + pd.Timedelta(hours=20)),
        })

    def avail(self, season, rnd, code, status="a", chance=None, hours_before_cutoff=3.0, element=1):
        cutoff = deadline(season, rnd) - pd.Timedelta(hours=spec.CUTOFF_HOURS_BEFORE_DEADLINE)
        taken = cutoff - pd.Timedelta(hours=hours_before_cutoff)
        self.availability.append({
            "season": season, "gameweek": rnd, "snapshot_timestamp": taken.strftime("%Y%m%d%H%M%S"),
            "snapshot_age_hours": hours_before_cutoff, "code": code, "id": element, "status": status,
            "chance_of_playing_next_round": chance, "chance_of_playing_this_round": chance,
            "news": "", "news_added": None,
        })

    def fill_regular_season(self, season, rounds, code_base=1000, n=8):
        """n players on two teams, a mix of regular starters and squad players."""
        for i in range(n):
            element, code = i + 1, code_base + i
            self.player(season, element, code)
            team = TEAMS[i % 2][2]
            for r in range(1, rounds + 1):
                started = (i < 4) or (r % 3 == i % 3)
                minutes = 90 if (i < 2 and started) else (70 if started else (15 if r % 2 else 0))
                self.appearance(season, element, r, team, minutes, started)
                self.avail(season, r, code, status="a" if (r + i) % 7 else "d", chance=None if (r + i) % 7 else 50)

    # -- writing -----------------------------------------------------------
    def write(self):
        for season, rows in self.gw_rows.items():
            d = os.path.join(self.data_dir, "vaastav", season, "gws")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(d, "merged_gw.csv"), index=False)
            pd.DataFrame([{"id": e, "code": c, "element_type": t} for e, (c, t) in self.players[season].items()]
                         ).to_csv(os.path.join(self.data_dir, "vaastav", season, "players_raw.csv"), index=False)
            pd.DataFrame([{"id": i, "code": c, "name": n} for i, c, n in TEAMS]).to_csv(
                os.path.join(self.data_dir, "vaastav", season, "teams.csv"), index=False)
        avail = pd.DataFrame(self.availability, columns=AVAILABILITY_COLUMNS)
        os.makedirs(os.path.join(self.data_dir, "availability"), exist_ok=True)
        hist = avail[avail["season"] != spec.PROSPECTIVE_SEASON]
        hist.to_csv(os.path.join(self.data_dir, "availability", "historical_availability.csv"), index=False)
        cur = avail[avail["season"] == spec.PROSPECTIVE_SEASON]
        for name, rounds in [("gw1_gw2_recovered_availability.csv", [1, 2]), ("gw3_gw4_availability.csv", [3, 4])]:
            part = cur[cur["gameweek"].isin(rounds)]
            part.to_csv(os.path.join(self.data_dir, "availability", name), index=False)
        rows = [{"season": s, "gameweek": r, "deadline": deadline(s, r).strftime("%Y-%m-%dT%H:%M:%S")}
                for s in spec.HISTORICAL_SEASONS for r in range(1, 39)]
        os.makedirs(os.path.join(self.data_dir, "deadlines"), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(self.data_dir, "deadlines", "gameweek_deadlines.csv"), index=False)

    def db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(derived.BASE_SCHEMA)
        return conn

    def bootstrap(self, finished_through):
        events = [{"id": r, "deadline_time": deadline(spec.PROSPECTIVE_SEASON, r).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "finished": r <= finished_through} for r in range(1, 39)]
        return {"events": events}


def add_current_gw(conn, code, rnd, team_code, minutes, started):
    conn.execute("INSERT INTO player_gameweek_stats (code, season, round, team_code, minutes, starts) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (code, spec.PROSPECTIVE_SEASON, rnd, team_code, minutes, int(started)))


def add_snapshot(conn, code, rnd, fetched_at, status="a", chance=None, team_code=101, element_type=3):
    conn.execute("INSERT INTO player_availability_snapshots (code, fetched_at, season, next_gw, status, "
                 "chance_of_playing_next_round, team_code, element_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (code, pd.Timestamp(fetched_at).strftime("%Y%m%dT%H%M%SZ"), spec.PROSPECTIVE_SEASON, rnd,
                  status, chance, team_code, element_type))


@pytest.fixture
def world(tmp_path):
    return SyntheticWorld(str(tmp_path))


def training_frame(n=600, seed=0, seasons=spec.TRAINING_SEASONS):
    """A synthetic model frame with every raw feature, for model-level tests."""
    rng = np.random.default_rng(seed)
    statuses = ["available"] * 6 + spec.CATEGORICAL["availability_status"][1]
    roles = ["did_not_play"] + spec.CATEGORICAL["last_gw_role"][1]
    df = pd.DataFrame({
        "season": rng.choice(list(seasons), n),
        "round": rng.integers(1, 39, n),
        "code": rng.integers(1, 200, n),
        "availability_status": rng.choice(statuses, n),
        "last_gw_role": rng.choice(roles, n),
        "minutes_prior_3_gws": rng.integers(0, 271, n).astype(float),
        "current_season_start_rate": rng.random(n),
        "previous_season_start_rate": rng.random(n),
        "no_previous_season": rng.integers(0, 2, n),
        "first_game_at_club": rng.integers(0, 2, n),
    })
    df.loc[df["no_previous_season"] == 1, "previous_season_start_rate"] = np.nan
    df.loc[rng.random(n) < 0.05, "current_season_start_rate"] = np.nan
    z = (-1 + 2 * (df["last_gw_role"] == "started_60_plus") + 1.5 * df["current_season_start_rate"].fillna(0.5)
         - 3 * df["availability_status"].isin(["injured", "suspended", "unavailable"]))
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df
