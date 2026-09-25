"""Player start history and prediction snapshots, shared by scoring and
by consuming projects.

- Prior-season history from the community archive
  (vaastav/Fantasy-Premier-League), since the live API holds the current
  season only (docs/build_spec_p_starts.md Section 2.3); current-season
  history from this project's own derived.db.
- `build_xseason_features` builds `prev`/`roll4` continuously across the
  season boundary, resetting at every confirmed change of club;
  `next_period_features` gives each player's latest features for the next
  gameweek. scoring.py's persistence and season-rate baselines use these.
- `snapshot_predictions` writes a write-once, timestamped predictions file
  in the format derived.py loads.

These helpers used to live in starts_model.py alongside the lookup-table
P(start) models (`raw_lookup`, `refined_availability`), which were retired
when the interpretable logistic model (fpl_starts.ml) replaced them.
"""

import io
import json
import os
import urllib.request

import pandas as pd

from . import archiver
from .config import PREDICTIONS_DIR

COMMUNITY_ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)


def fetch_community_archive(path):
    """GET one file from the community archive and return its raw bytes.
    The only network seam -- tests inject a fake instead.
    """
    req = urllib.request.Request(
        COMMUNITY_ARCHIVE_BASE + "/" + path, headers={"User-Agent": "curl"}
    )
    return urllib.request.urlopen(req, timeout=90).read()


def load_prior_season_starts(fetch, season):
    """(code, GW, y, team) for every player-gameweek of one full completed
    season, from the community archive. Joined on `code`, not the raw
    `element` id `merged_gw.csv` uses directly -- ids are only stable
    within a season (the same problem this project's own archiver/derived
    layer solves for the live API; `players_raw.csv` carries both per
    season).

    `team` is `merged_gw.csv`'s own per-gameweek club name (confirmed live:
    e.g. "Arsenal", stable prose, not a season-relative numeric id the way
    `element` is) -- see build_xseason_features for why this needs to be
    carried alongside `y` rather than joined some other way.
    """
    gw = pd.read_csv(io.BytesIO(fetch(season + "/gws/merged_gw.csv")), low_memory=False)
    players = pd.read_csv(io.BytesIO(fetch(season + "/players_raw.csv")))
    id_to_code = players.set_index("id")["code"]
    gw["code"] = gw["element"].map(id_to_code)
    gw_col = "GW" if "GW" in gw.columns else "round"
    gw = gw.rename(columns={gw_col: "GW"})
    gw["y"] = gw["starts"].astype(int)
    if "team" not in gw.columns:
        gw["team"] = None
    return gw[["code", "GW", "y", "team"]].dropna(subset=["code"])


def load_current_season_starts(conn, season):
    """(code, GW, y, team) for the season being predicted, from this
    project's own derived.db. `team` (a club name, via `teams.name`) is the
    same kind of value load_prior_season_starts carries -- see
    build_xseason_features for why.
    """
    df = pd.read_sql(
        "SELECT pgs.code, pgs.round AS GW, pgs.starts, t.name AS team "
        "FROM player_gameweek_stats pgs "
        "LEFT JOIN teams t ON t.code = pgs.team_code "
        "WHERE pgs.season = ?",
        conn, params=(season,),
    )
    df["y"] = df["starts"].astype(int)
    return df[["code", "GW", "y", "team"]]


def build_xseason_features(prior_df, current_df):
    """One continuous per-code sequence -- prior season's gameweeks as
    periods 1..N, current season's as N+1 onward -- with `prev`/`roll4`
    computed across the join. See the module docstring for why this beats
    leaving them undefined for early-season rows.

    Team changes and stale history (added 2026-09-07, validated in
    notebooks/fpl_starts_analysis.ipynb section 11): `prev`/`roll4` are
    reset at every change of `team`, not just computed per `code` -- a
    player who moved clubs has history that describes a different squad
    context, not a discontinuity in the same one. Each row is assigned a
    `team_stint` (increments whenever `team` is *known* on both this row
    and the immediately preceding one for that code, and the two differ --
    a missing `team` on either side isn't positive evidence of a move, so
    it's never treated as one; same "ambiguous isn't evidence of a
    problem" stance as derived.py's opponent_mismatch), and `prev`/`roll4`
    are computed within (code, team_stint) rather than `code` alone. The
    first game(s) after a confirmed transfer therefore get NaN prev/roll4
    -- correctly falling through to cold_start treatment
    (predict_gameweek) until the player has built up some history at the
    new club, rather than carrying their old club's form across the move.
    `team_stint` is kept in the returned frame, not dropped --
    next_period_features needs it to isolate the *current* stint when
    building live prediction features.
    """
    period_offset = int(prior_df["GW"].max()) if len(prior_df) else 0
    prior = prior_df.copy()
    prior["period"] = prior["GW"]
    current = current_df.copy()
    current["period"] = current["GW"] + period_offset

    combined = pd.concat([prior, current], ignore_index=True)
    combined = combined.sort_values(["code", "period"])

    prev_team = combined.groupby("code")["team"].shift(1)
    confirmed_change = (
        combined["team"].notna() & prev_team.notna() & (combined["team"] != prev_team)
    )
    combined["team_stint"] = confirmed_change.groupby(combined["code"]).cumsum()

    group_keys = ["code", "team_stint"]
    combined["prev"] = combined.groupby(group_keys)["y"].shift(1)
    combined["roll4"] = combined.groupby(group_keys)["y"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return combined


def next_period_features(combined, current_team=None):
    """For each code, the `prev`/`roll4` that apply to one more period
    appended right after their last observed one -- the actual features to
    predict the *next* gameweek with, not the features attached to their
    last played row (which describe the row before that one). Public
    because scoring.py reuses it to build the persistence baseline.

    Restricted to each code's most recent `team_stint` (see
    build_xseason_features) -- a player's last 1-4 games at a *previous*
    club aren't a valid "recent form" signal for their next game at a new
    one. A player with zero rows in their current stint (transferred but
    hasn't played there yet) simply has no rows to aggregate here, so
    `prev`/`roll4` come back NaN for them, same as any other code missing
    from this frame -- predict_gameweek already treats that as cold_start.

    `current_team`: optional {code: team} of each player's team *right
    now* (from derived.db's live players/teams, not historical
    play-by-play) -- catches the transfer that matters most and that
    build_xseason_features's within-history stint detection structurally
    can't: one that happened before any gameweek has been played for the
    new club yet, so the archived data's last row for that code is still
    the old club and no team_stint break exists in the history at all.
    When given, a code whose last archived row's team doesn't match
    current_team[code] has its entire trailing stint excluded (not just
    reset at a boundary that isn't there yet), so it drops out of the
    returned frame entirely -- callers merge this on `code`, so a missing
    code reads as NaN/cold_start the same way any other unmatched code
    does. A code current_team doesn't cover is left alone rather than
    guessed stale. Omitted by scoring.py's retrospective persistence
    baseline, deliberately -- there, "current" would mean "as of the round
    being scored", which the historical data already reflects by the time
    it's scored, not something a live players/teams lookup can answer for
    a past gameweek anyway.
    """
    ordered = combined.sort_values("period")
    last_stint = ordered.groupby("code")["team_stint"].transform("last")
    current_stint_rows = ordered[ordered["team_stint"] == last_stint]

    if current_team is not None:
        last_team = ordered.groupby("code")["team"].last()
        looked_up = last_team.index.to_series().map(current_team)
        stale_codes = set(last_team.index[looked_up.notna() & (looked_up != last_team)])
        current_stint_rows = current_stint_rows[
            ~current_stint_rows["code"].isin(stale_codes)
        ]

    grouped = current_stint_rows.groupby("code")["y"]
    prev = grouped.last()
    roll4 = grouped.apply(lambda s: s.tail(4).mean())
    return pd.DataFrame({"prev": prev, "roll4": roll4}).reset_index()


def snapshot_predictions(predictions, season, target_round, model_version,
                          base_dir=PREDICTIONS_DIR, clock=archiver.utcnow):
    """Write `predictions` (a DataFrame with at least code and p_start) to a
    write-once, timestamped JSON file.
    Refuses to overwrite an existing snapshot for the same
    season/round/model_version/timestamp, same as the raw archiver.

    `model_version` distinguishes different prediction methods for the same
    gameweek so derived.py
    keeps the latest of each separately and scoring.py can compare them.
    """
    fetched_at = clock()
    filename = "gw{0:02d}_{1}_{2}.json".format(
        target_round, model_version, archiver.format_timestamp(fetched_at)
    )
    path = os.path.join(base_dir, season, filename)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.exists(path):
        raise archiver.ArchiveError(
            "refusing to overwrite existing predictions snapshot: {0}".format(path)
        )
    payload = {
        "season": season,
        "target_round": target_round,
        "model_version": model_version,
        "predicted_at": archiver.format_timestamp(fetched_at),
        "predictions": predictions.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
