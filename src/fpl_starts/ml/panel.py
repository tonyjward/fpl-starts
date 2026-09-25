"""The modelling panel: one row per player per fixture, with the six
predictors computed from information available before that gameweek.

Construction rules:

- Features are computed on a *gameweek-level* series (one row per player per
  gameweek, collapsing double gameweeks), always from gameweeks strictly
  before the one being described, and then broadcast back onto every fixture
  row of that gameweek. Both fixtures of a double gameweek are predicted at
  the same cutoff, so they get byte-identical features -- one fixture's
  outcome can never leak into the other's.
- Recent-form and current-season features reset when a player changes club
  (a confirmed change of `team_code` between consecutive gameweeks);
  previous-season role carries across a transfer.
- Availability is the latest snapshot strictly before the gameweek's
  prediction cutoff (deadline - 2h); anything later is rejected loudly.
- `stratum` (Core/Rotation/Marginal/Deep) is an evaluation label only, never
  a model input, computed from the same season's earlier gameweeks with the
  thresholds in `fpl_starts.scoring.label_strata`.
"""

import numpy as np
import pandas as pd

from . import data as mldata
from . import spec

PERIOD_FEATURES = [
    "team_stint", "first_game_at_club", "prev_started", "minutes_prev",
    "minutes_prior_3_gws", "current_season_start_rate",
    "previous_season_start_rate", "no_previous_season", "stratum",
]


# --- Fixture-level rows -------------------------------------------------------

def load_historical_rows(data_dir, seasons=spec.HISTORICAL_SEASONS):
    return pd.concat([mldata.load_vaastav_season(data_dir, s) for s in seasons], ignore_index=True)


def load_current_season_rows(conn, season, before_round):
    """Played gameweeks of the current season from db/derived.db, strictly
    before `before_round`. This archive is gameweek-level (no fixture id), so
    a gameweek is one row per player."""
    df = pd.read_sql(
        "SELECT code, round, team_code, minutes, starts FROM player_gameweek_stats "
        "WHERE season = ? AND round < ?", conn, params=(season, int(before_round)))
    return pd.DataFrame({
        "code": df["code"].astype(int), "season": season, "round": df["round"].astype(int),
        "fixture": np.nan, "team_code": df["team_code"], "minutes": df["minutes"].astype(int),
        "y": df["starts"].clip(upper=1).astype(int),
    })


def assign_periods(rows, season_order):
    """`period`: a gameweek index continuous across seasons, so "previous
    gameweek" at a season's GW1 is the prior season's final gameweek."""
    rows = rows.copy()
    offsets, offset = {}, 0
    for season in season_order:
        offsets[season] = offset
        in_season = rows.loc[rows["season"] == season, "round"]
        offset += int(in_season.max()) if len(in_season) else 38
    rows["period"] = rows["round"] + rows["season"].map(offsets)
    return rows


# --- Gameweek-level features ------------------------------------------------

def build_period_level(rows):
    """One row per (code, season, round): started if he started any fixture
    that gameweek, minutes summed across them."""
    agg = rows.groupby(["code", "season", "round", "period"], sort=False).agg(
        period_started=("y", "max"),
        period_minutes=("minutes", "sum"),
        team_code=("team_code", "first"),
    ).reset_index()
    return agg.sort_values(["code", "period"]).reset_index(drop=True)


def _stratum(starts_before, games_before):
    if pd.isna(games_before) or games_before == 0 or starts_before == 0:
        return "Deep"
    rate = starts_before / games_before
    if rate >= 0.50:
        return "Core"
    if rate >= 0.15:
        return "Rotation"
    return "Marginal"


def add_period_features(period):
    df = period.copy()

    prev_team = df.groupby("code")["team_code"].shift(1)
    changed = df["team_code"].notna() & prev_team.notna() & (df["team_code"] != prev_team)
    df["team_stint"] = changed.groupby(df["code"]).cumsum()

    stint = df.groupby(["code", "team_stint"], sort=False)
    df["prev_started"] = stint["period_started"].shift(1)
    df["minutes_prev"] = stint["period_minutes"].shift(1)
    df["first_game_at_club"] = df["prev_started"].isna().astype(int)
    # Minutes in the three gameweeks before last at this club: 0 when he has
    # only one earlier gameweek here, missing only on his first gameweek.
    df["minutes_prior_3_gws"] = stint["period_minutes"].transform(
        lambda s: s.shift(2).rolling(3, min_periods=1).sum())
    df.loc[df["minutes_prior_3_gws"].isna() & (df["first_game_at_club"] == 0), "minutes_prior_3_gws"] = 0.0

    season_stint = df.groupby(["code", "season", "team_stint"], sort=False)["period_started"]
    df["current_season_start_rate"] = season_stint.transform(lambda s: s.shift(1).expanding().mean())

    season_order = sorted(df["season"].unique())
    totals = df.groupby(["code", "season"])["period_started"].agg(["sum", "count"]).reset_index()
    totals["rate"] = totals["sum"] / totals["count"]
    df["previous_season_start_rate"] = np.nan
    df["no_previous_season"] = 1
    for prior, this in zip(season_order, season_order[1:]):
        prior_rate = totals[totals["season"] == prior].set_index("code")["rate"]
        mask = df["season"] == this
        df.loc[mask, "previous_season_start_rate"] = df.loc[mask, "code"].map(prior_rate)
        df.loc[mask, "no_previous_season"] = (~df.loc[mask, "code"].isin(prior_rate.index)).astype(int)

    season_grp = df.groupby(["code", "season"], sort=False)["period_started"]
    starts_before = season_grp.transform(lambda s: s.shift(1).expanding().sum())
    games_before = season_grp.transform(lambda s: s.shift(1).expanding().count())
    df["stratum"] = [_stratum(s, g) for s, g in zip(starts_before, games_before)]
    return df


def broadcast_to_fixtures(rows, period_features):
    keys = ["code", "season", "round", "period"]
    return rows.merge(period_features[keys + PERIOD_FEATURES], on=keys, how="left")


def last_gw_role(prev_started, minutes_prev):
    """Encode the previous gameweek at the current club. On a player's first
    gameweek at a club there is no previous one; that row takes the reference
    level and `first_game_at_club` carries the effect."""
    role = np.full(len(prev_started), "did_not_play", dtype=object)
    started = np.asarray(prev_started, dtype=float)
    minutes = np.asarray(minutes_prev, dtype=float)
    role[(started == 0) & (minutes > 0)] = "sub_appearance"
    role[(started == 1) & (minutes < 60)] = "started_under_60"
    role[(started == 1) & (minutes >= 60)] = "started_60_plus"
    return role


# --- Availability ----------------------------------------------------------

def availability_status(status, chance):
    """Map FPL's `status` + `chance_of_playing_next_round` to the model's
    fixed levels. Doubtful players are graded by chance of playing; a missing
    status means no availability record before the cutoff."""
    out = []
    for s, c in zip(status, chance):
        if s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s):
            out.append("unknown")
        elif s == "a":
            out.append("available")
        elif s == "d":
            if pd.notna(c) and c <= 25:
                out.append("doubtful_25")
            elif pd.notna(c) and c >= 75:
                out.append("doubtful_75")
            else:
                out.append("doubtful_50")
        elif s == "i":
            out.append("injured")
        elif s == "s":
            out.append("suspended")
        elif s in ("u", "n"):
            out.append("unavailable")
        else:
            raise ValueError("unrecognised FPL availability status: {0!r}".format(s))
    return out


def current_season_cutoffs(bootstrap, season, hours_before=spec.CUTOFF_HOURS_BEFORE_DEADLINE):
    """{(season, round): cutoff} from bootstrap-static's `events` schedule."""
    out = {}
    for event in bootstrap["events"]:
        deadline = pd.Timestamp(event["deadline_time"]).tz_convert(None)
        out[(season, int(event["id"]))] = deadline - pd.Timedelta(hours=hours_before)
    return out


def load_current_season_availability(conn, data_dir, season, rounds, cutoffs):
    """Pre-cutoff availability for the requested current-season rounds:
    the local files for the gameweeks this project's own archive predates,
    db/derived.db's archived snapshots for the rest. Raises if any requested
    round has no availability at all -- never silently predicts without it."""
    files = mldata.load_current_season_availability_files(data_dir)
    files = files[(files["season"] == season) & files["round"].isin(rounds)]
    file_rounds = set(files["round"].unique())

    snaps = pd.read_sql(
        "SELECT code, fetched_at, next_gw AS round, status, chance_of_playing_next_round, "
        "team_code, element_type FROM player_availability_snapshots WHERE season = ?",
        conn, params=(season,))
    snaps = snaps[snaps["round"].isin([r for r in rounds if r not in file_rounds])]
    snaps = snaps[snaps["element_type"] != mldata.MANAGER_ELEMENT_TYPE]
    snaps["snapshot_at"] = pd.to_datetime(snaps["fetched_at"], format="%Y%m%dT%H%M%SZ")
    snaps["season"] = season
    snaps["cutoff"] = pd.Series([cutoffs.get((season, int(r))) for r in snaps["round"]],
                                index=snaps.index, dtype="datetime64[ns]")
    snaps = snaps[snaps["snapshot_at"] < snaps["cutoff"]]
    latest = snaps.sort_values("snapshot_at").groupby(["round", "code"]).tail(1)

    cols = ["season", "round", "code", "status", "chance_of_playing_next_round", "snapshot_at"]
    out = pd.concat([files[cols], latest[cols + ["team_code"]]], ignore_index=True)
    missing = sorted(set(rounds) - set(out["round"].unique()))
    if missing:
        raise mldata.MissingLocalDataError(
            "no pre-cutoff availability for {0} gameweek(s) {1}".format(season, missing))
    return mldata.check_pre_cutoff(out, cutoffs)


def attach_availability(rows, availability):
    keys = ["season", "round", "code"]
    avail = availability[keys + ["status", "chance_of_playing_next_round", "snapshot_at"]]
    if avail.duplicated(keys).any():
        raise ValueError("more than one availability row per (season, round, code)")
    merged = rows.merge(avail, on=keys, how="left")
    merged["availability_status"] = availability_status(
        merged["status"].tolist(), merged["chance_of_playing_next_round"].tolist())
    return merged


# --- Assembled panels --------------------------------------------------------

def with_features(rows, season_order):
    rows = assign_periods(rows, season_order)
    period = add_period_features(build_period_level(rows))
    panel = broadcast_to_fixtures(rows, period)
    panel["last_gw_role"] = last_gw_role(panel["prev_started"], panel["minutes_prev"])
    return panel


def build_historical_panel(data_dir):
    """Every historical season, features and pre-cutoff availability
    attached. Includes 2022-23 (context-only) rows; callers select the rows
    they train or evaluate on."""
    rows = load_historical_rows(data_dir)
    panel = with_features(rows, list(spec.HISTORICAL_SEASONS))
    availability = mldata.load_historical_availability(data_dir)
    availability = availability[availability["season"].isin(spec.HISTORICAL_SEASONS)]
    mldata.check_pre_cutoff(availability, mldata.load_historical_cutoffs(
        data_dir, spec.CUTOFF_HOURS_BEFORE_DEADLINE))
    return attach_availability(panel, availability)
