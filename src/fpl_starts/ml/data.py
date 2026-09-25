"""Local-only historical inputs under `data/` (see config.DATA_DIR).

These files are intentionally not distributed with this repository. Every
loader here fails loudly when one is missing, rather than silently training
or predicting without availability information.

Expected layout, relative to the data directory:

    vaastav/<season>/gws/merged_gw.csv
    vaastav/<season>/players_raw.csv
    vaastav/<season>/teams.csv
    availability/historical_availability.csv        seasons before 2026-27
    availability/gw1_gw2_recovered_availability.csv 2026-27 GW1-2
    availability/gw3_gw4_availability.csv           2026-27 GW3-4
    deadlines/gameweek_deadlines.csv                historical deadlines

Availability files share one schema: one row per (season, gameweek, code),
the latest snapshot taken before that gameweek's prediction cutoff.
"""

import os

import numpy as np
import pandas as pd

HISTORICAL_AVAILABILITY = os.path.join("availability", "historical_availability.csv")
CURRENT_SEASON_AVAILABILITY = [
    os.path.join("availability", "gw1_gw2_recovered_availability.csv"),
    os.path.join("availability", "gw3_gw4_availability.csv"),
]
DEADLINES = os.path.join("deadlines", "gameweek_deadlines.csv")

MANAGER_ELEMENT_TYPE = 5  # FPL "Manager" entries (2024-25 onwards): not players


class MissingLocalDataError(FileNotFoundError):
    """A required local-only input under data/ is absent."""


def require(data_dir, relative_path):
    path = os.path.join(data_dir, relative_path)
    if not os.path.isfile(path):
        raise MissingLocalDataError(
            "required local data file not found: {0}\n"
            "The logistic P(start) model needs historical inputs under {1}/ that "
            "are intentionally not distributed with this repository (see "
            "docs/logistic_p_start_model.md). Nothing was trained or predicted."
            .format(path, data_dir)
        )
    return path


def load_vaastav_season(data_dir, season):
    """Fixture-level player rows for one historical season: one row per
    (code, fixture), double gameweeks preserved.

    - players are identified by `code` (stable across seasons), never by the
      season-local `element` id;
    - FPL Manager entries (element_type 5) are dropped -- they are not players;
    - `team_code` is the player's club *for that fixture* (the row's own team
      name), not his end-of-season club, so pre-transfer rows keep the club he
      actually played for;
    - exact duplicate (code, round, fixture) rows in the source are dropped.
    """
    gw = pd.read_csv(require(data_dir, os.path.join("vaastav", season, "gws", "merged_gw.csv")),
                     low_memory=False)
    players = pd.read_csv(require(data_dir, os.path.join("vaastav", season, "players_raw.csv")))
    teams = pd.read_csv(require(data_dir, os.path.join("vaastav", season, "teams.csv")))

    id_to_code = players.set_index("id")["code"]
    element_type = players.set_index("id")["element_type"]
    team_name_to_code = teams.set_index("name")["code"]

    gw = gw[gw["element"].map(element_type) != MANAGER_ELEMENT_TYPE].copy()
    if "GW" in gw.columns:
        gw = gw.drop(columns=[c for c in ["round"] if c in gw.columns]).rename(columns={"GW": "round"})

    out = pd.DataFrame({
        "code": gw["element"].map(id_to_code),
        "season": season,
        "round": gw["round"].astype(int),
        "fixture": gw["fixture"],
        "team_code": gw["team"].map(team_name_to_code),
        "minutes": gw["minutes"].astype(int),
        "y": gw["starts"].astype(int).clip(upper=1),
    })
    out = out.dropna(subset=["code"])
    out["code"] = out["code"].astype(int)
    return out.drop_duplicates(subset=["code", "round", "fixture"], keep="first").reset_index(drop=True)


def _load_availability_csv(path):
    df = pd.read_csv(path, dtype={"snapshot_timestamp": str})
    return pd.DataFrame({
        "season": df["season"],
        "round": df["gameweek"].astype(int),
        "code": df["code"].astype(int),
        "status": df["status"],
        "chance_of_playing_next_round": df["chance_of_playing_next_round"],
        "snapshot_at": pd.to_datetime(df["snapshot_timestamp"], format="%Y%m%d%H%M%S"),
    })


def load_historical_availability(data_dir):
    return _load_availability_csv(require(data_dir, HISTORICAL_AVAILABILITY))


def load_current_season_availability_files(data_dir):
    return pd.concat([_load_availability_csv(require(data_dir, p)) for p in CURRENT_SEASON_AVAILABILITY],
                     ignore_index=True)


def load_historical_cutoffs(data_dir, hours_before=2):
    """{(season, round): prediction cutoff (naive UTC)} for historical seasons."""
    df = pd.read_csv(require(data_dir, DEADLINES))
    deadline = pd.to_datetime(df["deadline"])
    cutoff = deadline - pd.Timedelta(hours=hours_before)
    return {(s, int(r)): c for s, r, c in zip(df["season"], df["gameweek"], cutoff)}


def check_pre_cutoff(availability, cutoffs):
    """Raise if any availability row is not strictly before its gameweek's
    prediction cutoff, or has no known cutoff to be checked against."""
    keys = list(zip(availability["season"], availability["round"]))
    cut = pd.Series([cutoffs.get(k) for k in keys], index=availability.index, dtype="datetime64[ns]")
    unknown = cut.isna()
    if unknown.any():
        missing = sorted(set(k for k, u in zip(keys, unknown) if u))
        raise ValueError("no prediction cutoff known for {0}".format(missing[:5]))
    late = ~(availability["snapshot_at"].values < cut.values)
    if np.any(late):
        raise ValueError("{0} availability rows are not strictly before their prediction cutoff"
                         .format(int(late.sum())))
    return availability
