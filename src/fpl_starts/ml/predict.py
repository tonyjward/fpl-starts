"""Score a prospective-season gameweek with the frozen logistic model.

Never fits anything: it loads the frozen model.json and applies it. Features
for gameweek N use only information available before N's prediction cutoff
(deadline - 2h):

- outcomes: this season's gameweeks strictly before N (all of which must be
  finished), plus the historical seasons;
- availability: the latest snapshot strictly before the cutoff -- local
  files for the gameweeks this project's own archive predates, archived
  snapshots in db/derived.db for the rest;
- the players scored are those in that pre-cutoff availability snapshot.

Each run writes a write-once snapshot to predictions/<season>/, in the same
format as the other models (so `fpl-starts-derive` loads it and
`fpl-starts-score` can score it), extended with a full per-player
explanation: raw value, transformed value, coefficient and log-odds
contribution for every model feature, plus intercept, logit and p_start.
A snapshot generated after the gameweek's deadline is marked as such: its
inputs are still strictly pre-cutoff and the model still never saw the
season, but it was not a real-time forecast.

Usage: fpl-starts-logistic-predict --target-round N [N ...]
"""

import argparse
import json
import os
import sqlite3

import pandas as pd

from .. import archiver, config, derived
from . import logistic, panel as mlpanel, spec
from .data import MissingLocalDataError


def _pool(conn, season, target_round, availability, history):
    """Players to score and their club for gameweek N: the club in the
    pre-cutoff snapshot where recorded, else the club listed for that
    gameweek in this season's archive, else the last club on record."""
    pool = availability[["code"]].drop_duplicates().copy()
    team = {}
    if "team_code" in availability.columns:
        team.update(availability.dropna(subset=["team_code"]).set_index("code")["team_code"].to_dict())
    listed = pd.read_sql("SELECT code, team_code FROM player_gameweek_stats WHERE season = ? AND round = ?",
                         conn, params=(season, int(target_round)))
    last = history.sort_values(["season", "round"]).groupby("code")["team_code"].last()
    for source in (listed.set_index("code")["team_code"].to_dict(), last.to_dict()):
        for code, t in source.items():
            team.setdefault(code, t)
    pool["team_code"] = pool["code"].map(team)
    return pool


def build_target_rows(conn, data_dir, raw_dir, season, target_round):
    if season != spec.PROSPECTIVE_SEASON:
        raise ValueError("this model scores {0} only".format(spec.PROSPECTIVE_SEASON))
    bootstrap = derived.latest_bootstrap_payload(raw_dir, season)
    if bootstrap is None:
        raise MissingLocalDataError("no archived bootstrap-static for {0} under {1}".format(season, raw_dir))
    events = {int(e["id"]): e for e in bootstrap["events"]}
    unfinished = [r for r in range(1, target_round) if not events[r]["finished"]]
    if unfinished:
        raise ValueError("gameweek(s) {0} before GW{1} are not finished yet".format(unfinished, target_round))
    cutoffs = mlpanel.current_season_cutoffs(bootstrap, season)

    history = pd.concat([mlpanel.load_historical_rows(data_dir),
                         mlpanel.load_current_season_rows(conn, season, target_round)], ignore_index=True)
    availability = mlpanel.load_current_season_availability(conn, data_dir, season, [target_round], cutoffs)
    pool = _pool(conn, season, target_round, availability, history)
    stub = pd.DataFrame({"code": pool["code"].astype(int), "season": season, "round": int(target_round),
                         "fixture": float("nan"), "team_code": pool["team_code"], "minutes": 0,
                         "y": float("nan")})
    panel = mlpanel.with_features(pd.concat([history, stub], ignore_index=True),
                                  list(spec.HISTORICAL_SEASONS) + [season])
    target = panel[(panel["season"] == season) & (panel["round"] == target_round)]
    target = mlpanel.attach_availability(target, availability).reset_index(drop=True)
    return target, cutoffs[(season, target_round)], events[target_round]["deadline_time"]


def predict_round(conn, data_dir, raw_dir, model, season, target_round):
    rows, cutoff, deadline = build_target_rows(conn, data_dir, raw_dir, season, target_round)
    explanations = model.explain(rows)
    records, detail = [], []
    for row, exp in zip(rows.itertuples(index=False), explanations):
        records.append({
            "code": int(row.code), "p_start": exp["p_start"], "logit": exp["logit"],
            "cold_start": bool(row.first_game_at_club), "n_observed": None, "method": spec.MODEL_ID,
            "team_code": None if pd.isna(row.team_code) else int(row.team_code),
            "availability_status": row.availability_status, "last_gw_role": row.last_gw_role,
        })
        detail.append(dict(code=int(row.code), **exp))
    return records, detail, cutoff, deadline


def _naive_utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def write_snapshot(records, detail, model, model_path, season, target_round, cutoff, deadline,
                   base_dir=config.PREDICTIONS_DIR, clock=archiver.utcnow):
    predicted_at = clock()
    stamp = archiver.format_timestamp(predicted_at)
    path = os.path.join(base_dir, season, "gw{0:02d}_{1}_{2}.json".format(target_round, spec.MODEL_VERSION, stamp))
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite existing predictions snapshot: {0}".format(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    deadline_ts = _naive_utc(deadline)
    payload = {
        "season": season, "target_round": int(target_round), "model_version": spec.MODEL_VERSION,
        "predicted_at": stamp,
        "prediction_cutoff": cutoff.isoformat() + "Z",
        "deadline": deadline,
        "generated_after_deadline": bool(_naive_utc(predicted_at) >= deadline_ts),
        "model": {"model_id": spec.MODEL_ID, "model_sha256": logistic.file_sha256(model_path),
                  "created_at": model.metadata.get("created_at"),
                  "training_seasons": model.metadata.get("training_seasons"),
                  "intercept": model.intercept, "C": model.C},
        "predictions": records,
        "explanations": detail,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    return path


def _main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target-round", type=int, nargs="+", required=True)
    parser.add_argument("--season", default=spec.PROSPECTIVE_SEASON)
    parser.add_argument("--db-path", default=config.DERIVED_DB_PATH)
    parser.add_argument("--base-dir", default=config.RAW_DIR, help="raw archive directory")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    parser.add_argument("--predictions-dir", default=config.PREDICTIONS_DIR)
    args = parser.parse_args()

    directory = logistic.model_dir(args.models_dir)
    model = logistic.load(directory)
    if not os.path.isfile(args.db_path):
        raise SystemExit("no database at {0} -- run fpl-starts-derive first".format(args.db_path))
    conn = sqlite3.connect(args.db_path)
    try:
        for target_round in args.target_round:
            records, detail, cutoff, deadline = predict_round(conn, args.data_dir, args.base_dir, model,
                                                              args.season, target_round)
            path = write_snapshot(records, detail, model, os.path.join(directory, "model.json"), args.season,
                                  target_round, cutoff, deadline, base_dir=args.predictions_dir)
            p = pd.Series([r["p_start"] for r in records])
            print("GW{0}: {1} players, mean p_start {2:.3f} -> {3}".format(target_round, len(records), p.mean(), path))
    finally:
        conn.close()


if __name__ == "__main__":
    _main()
