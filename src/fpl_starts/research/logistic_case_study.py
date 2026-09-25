"""Aggregate analyses and loaders behind the public logistic P(start) case
study, `notebooks/logistic_p_start_model.ipynb`.

This lives outside `fpl_starts.ml` on purpose. The frozen model records a
hash of every source file in that package (`model_source_sha256`), so adding
research code there would make the frozen artefact look as if it had been
fitted from different code.

Rules this module keeps:

- Development seasons only (2023-24 to 2025-26). Nothing reads a 2026-27
  outcome.
- Comparison models are fitted only inside the walk-forward folds of
  `fpl_starts.ml.evaluate`, with the specification swapped in memory by
  `specification()`. `spec.py` and the frozen model are never touched.
- Output is aggregate: correlations, VIFs, metrics and coefficients. No
  player rows and no availability records.

Usage: fpl-starts-logistic-case-study [--data-dir data] [--output PATH]
"""

import argparse
import contextlib
import glob
import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd

from .. import config
from ..ml import evaluate, logistic, panel as mlpanel, spec
from ..ml.preprocessing import Preprocessor, model_frame

CASE_STUDY_PATH = os.path.join(config.MODELS_DIR, spec.MODEL_VERSION, "case_study.json")

# --- Candidate features -------------------------------------------------------
#
# The wider set of overlapping role and availability measures the compact
# model was simplified from. Each is computed from gameweeks strictly before
# the one it describes, within the player's spell at his current club, like
# the model's own features.

CANDIDATES = {
    "prev_started": "started his previous gameweek",
    "consecutive_starts": "run of consecutive starts up to last gameweek",
    "starts_last_2": "starts in the last 2 gameweeks",
    "starts_last_4": "starts in the last 4 gameweeks",
    "starts_last_10": "starts in the last 10 gameweeks",
    "start_rate_last_4": "share of the last 4 gameweeks started",
    "start_rate_last_10": "share of the last 10 gameweeks started",
    "minutes_last_1": "minutes last gameweek",
    "minutes_last_2": "minutes in the last 2 gameweeks",
    "minutes_last_4": "minutes in the last 4 gameweeks",
    "minutes_last_10": "minutes in the last 10 gameweeks",
    "minutes_prior_3_gws": "minutes in the 3 gameweeks before last",
    "current_season_starts": "starts so far this season",
    "current_season_start_rate": "share of this season's gameweeks started",
    "previous_season_starts": "starts last season",
    "previous_season_minutes": "minutes last season",
    "previous_season_start_rate": "share of last season's gameweeks started",
    "chance_of_playing": "FPL chance of playing next round (100 when available)",
    "is_available": "FPL status is available",
}

# Pairs the case study calls out as measuring the same thing.
HIGHLIGHTED_PAIRS = [
    ("starts_last_4", "start_rate_last_4"),
    ("minutes_last_2", "minutes_last_4"),
    ("minutes_last_4", "minutes_last_10"),
    ("previous_season_starts", "previous_season_start_rate"),
    ("previous_season_minutes", "previous_season_start_rate"),
    ("prev_started", "consecutive_starts"),
    ("chance_of_playing", "is_available"),
]

# The compact heatmap shows one or two members of each concept.
HEATMAP_FEATURES = [
    "prev_started", "consecutive_starts", "starts_last_4", "start_rate_last_4",
    "minutes_last_4", "minutes_last_10", "minutes_prior_3_gws",
    "current_season_start_rate", "previous_season_start_rate", "previous_season_minutes",
    "chance_of_playing", "is_available",
]


def _run_length(started):
    """Consecutive starts ending at each position (0 after a non-start)."""
    out, run = [], 0
    for s in started:
        run = run + 1 if s == 1 else 0
        out.append(run)
    return pd.Series(out, index=started.index)


def add_candidate_features(panel):
    """`panel` from `build_historical_panel`, plus the CANDIDATES columns.
    Computed on the gameweek-level series and broadcast back, so both
    fixtures of a double gameweek share them."""
    period = panel.groupby(["code", "season", "round", "period", "team_stint"], sort=False).agg(
        started=("y", "max"), minutes=("minutes", "sum")).reset_index()
    period = period.sort_values(["code", "period"]).reset_index(drop=True)

    stint = period.groupby(["code", "team_stint"], sort=False)
    for k in (2, 4, 10):
        period["starts_last_{0}".format(k)] = stint["started"].transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=1).sum())
        period["minutes_last_{0}".format(k)] = stint["minutes"].transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=1).sum())
    for k in (4, 10):
        period["start_rate_last_{0}".format(k)] = stint["started"].transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=1).mean())
    period["minutes_last_1"] = stint["minutes"].shift(1)
    period["consecutive_starts"] = stint["started"].transform(lambda s: _run_length(s).shift(1))
    period["current_season_starts"] = period.groupby(["code", "season", "team_stint"], sort=False)[
        "started"].transform(lambda s: s.shift(1).expanding().sum())

    totals = period.groupby(["code", "season"]).agg(starts=("started", "sum"), minutes=("minutes", "sum"))
    seasons = sorted(period["season"].unique())
    prior_of = dict(zip(seasons[1:], seasons))
    prior = pd.DataFrame({"code": period["code"], "season": period["season"].map(prior_of).astype(object)})
    prior = prior.merge(totals.reset_index(), on=["code", "season"], how="left")
    period["previous_season_starts"] = prior["starts"].values
    period["previous_season_minutes"] = prior["minutes"].values

    keys = ["code", "season", "round", "period", "team_stint"]
    new = [c for c in period.columns if c not in keys + ["started", "minutes"]]
    out = panel.merge(period[keys + new], on=keys, how="left")
    out["is_available"] = (out["availability_status"] == "available").astype(int)
    chance = pd.to_numeric(out["chance_of_playing_next_round"], errors="coerce")
    out["chance_of_playing"] = chance.where(chance.notna(), np.where(out["is_available"] == 1, 100.0, np.nan))
    return out


def development_rows(panel):
    return panel[panel["season"].isin(spec.TRAINING_SEASONS)]


def redundancy(panel):
    """Spearman correlations among the candidates on the development rows."""
    dev = development_rows(panel)
    corr = dev[list(CANDIDATES)].corr(method="spearman")
    return {
        "heatmap_features": HEATMAP_FEATURES,
        "heatmap": corr.loc[HEATMAP_FEATURES, HEATMAP_FEATURES].round(4).values.tolist(),
        "highlighted_pairs": [{"a": a, "b": b, "spearman": float(corr.loc[a, b])} for a, b in HIGHLIGHTED_PAIRS],
        "n_rows": int(len(dev)),
    }


# --- Specifications ------------------------------------------------------------

FROZEN = {"continuous": list(spec.CONTINUOUS), "binary": list(spec.BINARY),
          "categorical": dict(spec.CATEGORICAL)}

MINUTES_LAST_4 = {"continuous": ["minutes_last_4"] + FROZEN["continuous"][1:],
                  "binary": FROZEN["binary"], "categorical": FROZEN["categorical"]}

# Every candidate at once, plus the frozen model's own categorical and flag
# predictors: what "rely on regularisation" would look like.
FULL_CANDIDATE_SET = {
    "continuous": [c for c in CANDIDATES if c not in ("prev_started", "is_available")],
    "binary": ["prev_started_filled", "is_available", "no_previous_season", "first_game_at_club"],
    "categorical": FROZEN["categorical"],
}


@contextlib.contextmanager
def specification(definition):
    """Swap the predictors `fpl_starts.ml` reads from `spec`, in memory only,
    and restore them afterwards."""
    saved = spec.CONTINUOUS, spec.BINARY, spec.CATEGORICAL, spec.RAW_FEATURES
    spec.CONTINUOUS = list(definition["continuous"])
    spec.BINARY = list(definition["binary"])
    spec.CATEGORICAL = dict(definition["categorical"])
    spec.RAW_FEATURES = list(spec.CATEGORICAL) + spec.CONTINUOUS + spec.BINARY
    try:
        yield
    finally:
        spec.CONTINUOUS, spec.BINARY, spec.CATEGORICAL, spec.RAW_FEATURES = saved


def vif(panel):
    """Variance inflation factor of every model column under the current
    specification, on the development rows."""
    rows = development_rows(panel)
    X = Preprocessor().fit(model_frame(rows)).transform(model_frame(rows)).astype(float)
    X = X.loc[:, X.std() > 0]
    design = np.column_stack([np.ones(len(X)), X.values])
    out = {}
    for i, column in enumerate(X.columns, start=1):
        target = design[:, i]
        others = np.delete(design, i, axis=1)
        beta = np.linalg.lstsq(others, target, rcond=None)[0]
        residual = target - others @ beta
        r2 = 1 - residual @ residual / ((target - target.mean()) ** 2).sum()
        out[column] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out


def _summary(result):
    def block(m):
        return {"brier": m["brier"], "rotation_brier": m["by_stratum"]["Rotation"]["brier"],
                "calibration_slope": m["calibration"]["slope"], "ece": m["calibration"]["ece"]}
    windows = [("overall", result["overall"])] + list(result["by_test_season"].items())
    folds = result["folds"]
    names = list(folds[0]["coefficients"])
    signs_agree = all(np.sign(folds[0]["coefficients"][n]) == np.sign(folds[1]["coefficients"][n]) for n in names)
    return {
        "logistic": {w: block(b["logistic"]) for w, b in windows},
        "naive": {w: block(b["naive"]) for w, b in windows},
        "C_by_fold": {f["test_season"]: f["C"] for f in folds},
        "coefficients_by_fold": {f["test_season"]: f["coefficients"] for f in folds},
        "sign_flips_across_folds": [n for n in names
                                    if np.sign(folds[0]["coefficients"][n]) != np.sign(folds[1]["coefficients"][n])],
        "signs_agree_across_folds": bool(signs_agree),
        "n_coefficients": len(names),
    }


def evaluate_specification(panel, definition):
    with specification(definition):
        result, _ = evaluate.run(panel)
        vifs = vif(panel)
    summary = _summary(result)
    summary["vif"] = vifs
    summary["max_vif"] = max(vifs.values())
    return summary


# --- Performance by player type ---------------------------------------------------

BANDS = ["No gameweek yet this season", "Never started this season", "Started under 15%",
         "Started 15-50%", "Started 50-90%", "Started 90%+"]


def _season_so_far(panel):
    """starts and gameweeks so far this season, strictly before each
    gameweek -- the quantities `fpl_starts.ml.panel` builds strata from."""
    period = panel.groupby(["code", "season", "round"], sort=False)["y"].max().reset_index()
    period = period.sort_values(["code", "season", "round"])
    grp = period.groupby(["code", "season"], sort=False)["y"]
    period["starts_before"] = grp.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    period["games_before"] = grp.transform(lambda s: s.shift(1).expanding().count()).fillna(0)
    return period.drop(columns="y")


def band(starts_before, games_before):
    if games_before == 0:
        return BANDS[0]
    if starts_before == 0:
        return BANDS[1]
    rate = starts_before / games_before
    if rate < 0.15:
        return BANDS[2]
    if rate < 0.50:
        return BANDS[3]
    if rate < 0.90:
        return BANDS[4]
    return BANDS[5]


def _group_metrics(rows, key, order):
    total_n = len(rows)
    sq = {m: (rows[m] - rows["y"]) ** 2 for m in ("naive", "logistic")}
    out = []
    for level in order:
        m = rows[key] == level
        if not m.any():
            continue
        rate = float(rows.loc[m, "y"].mean())
        out.append({
            "group": level, "n": int(m.sum()), "share_of_rows": float(m.sum() / total_n),
            "start_rate": rate, "base_rate_brier": rate * (1 - rate),
            "brier_naive": float(sq["naive"][m].mean()), "brier_logistic": float(sq["logistic"][m].mean()),
            "share_of_squared_error_logistic": float(sq["logistic"][m].sum() / sq["logistic"].sum()),
        })
    return out


def performance_by_player_type(panel):
    """Out-of-fold predictions of the frozen specification on the two test
    seasons, broken down by stratum and by a finer start-rate band."""
    with specification(FROZEN):
        _, pred = evaluate.run(panel)
    pred = pred.merge(_season_so_far(panel), on=["code", "season", "round"], how="left")
    pred["band"] = [band(s, g) for s, g in zip(pred["starts_before"], pred["games_before"])]
    return {"n": int(len(pred)),
            "by_stratum": _group_metrics(pred, "stratum", evaluate.STRATA),
            "by_band": _group_metrics(pred, "band", BANDS)}


NAIVE_CELLS = [(1.0, "started last gameweek"), (0.0, "did not start last gameweek"),
               (-1.0, "first gameweek at club")]


def naive_lookup(panel):
    """The benchmark's lookup table in each fold: the training window's start
    rate in each cell, exactly as `evaluate.naive_baseline` fits it."""
    dev = development_rows(panel)
    out = {}
    for test_season, train_seasons in evaluate.FOLDS:
        train = dev[dev["season"].isin(train_seasons)]
        cells = train.groupby(train["prev_started"].fillna(-1))["y"].agg(["mean", "size"])
        out[test_season] = {"train_seasons": train_seasons,
                            "cells": [{"cell": label, "p_start": float(cells.loc[k, "mean"]),
                                       "n": int(cells.loc[k, "size"])}
                                      for k, label in NAIVE_CELLS if k in cells.index]}
    return out


def run(data_dir):
    panel = add_candidate_features(mlpanel.build_historical_panel(data_dir))
    panel["prev_started_filled"] = panel["prev_started"].fillna(0).astype(int)
    return {
        "note": "development seasons 2023-24 to 2025-26 only; walk-forward folds as in "
                "fpl_starts.ml.evaluate; no 2026-27 outcomes used",
        "candidates": CANDIDATES,
        "redundancy": redundancy(panel),
        "performance_by_player_type": performance_by_player_type(panel),
        "naive_lookup": naive_lookup(panel),
        "specifications": {
            "frozen": evaluate_specification(panel, FROZEN),
            "minutes_last_4": evaluate_specification(panel, MINUTES_LAST_4),
            "full_candidate_set": evaluate_specification(panel, FULL_CANDIDATE_SET),
        },
    }


# --- Loaders for the notebook --------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_snapshots(predictions_dir=config.PREDICTIONS_DIR, season=spec.PROSPECTIVE_SEASON,
                   model_version=spec.MODEL_VERSION):
    """{gameweek: snapshot}, the latest snapshot per gameweek."""
    out = {}
    pattern = os.path.join(predictions_dir, season, "gw*_{0}_*.json".format(model_version))
    for path in sorted(glob.glob(pattern)):
        snapshot = load_json(path)
        snapshot["path"] = path
        out[snapshot["target_round"]] = snapshot
    return out


def player_names(db_path=config.DERIVED_DB_PATH):
    """{code: (web_name, team short name)} from the current-season database."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT p.code, p.web_name, t.short_name FROM players p "
                            "LEFT JOIN teams t ON t.code = p.team_code").fetchall()
    return {code: (name, team) for code, name, team in rows}


def explanation(snapshot, code):
    """The stored explanation of one player's prediction."""
    for e in snapshot["explanations"]:
        if e["code"] == code:
            return e
    raise KeyError("player {0} not in gameweek {1}".format(code, snapshot["target_round"]))


def is_complete(e):
    """Every model column is present and the stored arithmetic adds up."""
    names = [f["feature"] for f in e["features"]]
    if names != spec.transformed_feature_names():
        return False
    logit = e["intercept"] + sum(f["contribution"] for f in e["features"])
    return math.isclose(logit, e["logit"], abs_tol=1e-9) and math.isclose(
        1 / (1 + math.exp(-logit)), e["p_start"], abs_tol=1e-9)


CONCEPTS = [
    ("Availability", ["availability_status"]),
    ("Last gameweek", ["last_gw_role"]),
    ("Recent minutes", ["minutes_prior_3_gws"]),
    ("Current-season role", ["current_season_start_rate"]),
    ("Previous-season role", ["previous_season_start_rate", "no_previous_season"]),
    ("New at club", ["first_game_at_club"]),
]


def contributions_by_concept(e):
    """Log-odds contribution of each football concept (categorical dummies
    and paired flags summed), in CONCEPTS order."""
    return {concept: sum(f["contribution"] for f in e["features"] if f["raw_feature"] in raw)
            for concept, raw in CONCEPTS}


def select_regular_starter(snapshots, names, preferred="Haaland"):
    """The regular-starter worked example: `preferred` in the latest
    gameweek where he is available, started 60+ minutes last gameweek, has a
    current-season start rate and a complete explanation. If no such
    prediction exists, the player meeting the same conditions with the
    highest P(start) in the latest gameweek (ties: lower code).

    Returns (gameweek, code)."""
    def eligible(snapshot):
        preds = {p["code"]: p for p in snapshot["predictions"]}
        for e in snapshot["explanations"]:
            p = preds[e["code"]]
            raw = {f["raw_feature"]: f["raw_value"] for f in e["features"]}
            if (p["availability_status"] == "available" and p["last_gw_role"] == "started_60_plus"
                    and raw["current_season_start_rate"] is not None and is_complete(e)):
                yield e
    latest_first = sorted(snapshots, reverse=True)
    for gw in latest_first:
        for e in eligible(snapshots[gw]):
            if names.get(e["code"], ("",))[0] == preferred:
                return gw, e["code"]
    gw = latest_first[0]
    best = min(eligible(snapshots[gw]), key=lambda e: (-e["p_start"], e["code"]))
    return gw, best["code"]


FLAG_PREFERENCE = ["doubtful_25", "doubtful_50", "doubtful_75", "injured", "suspended", "unavailable"]


def select_flagged_example(snapshots):
    """Deterministic choice of the availability-flagged worked example:

    1. availability status other than the reference (`available`) and not
       `unknown`;
    2. a complete explanation (`is_complete`);
    3. the first status in FLAG_PREFERENCE that any player has: a doubtful
       grading shows the model adjusting a probability, where a hard
       injury simply sends it near zero;
    4. within that status, the player whose other features alone give the
       highest P(start) -- the clearest "would normally start" case;
    5. ties: later gameweek, then lower player code.

    Returns (gameweek, code, status, p_without_availability)."""
    candidates = []
    for gw, snapshot in snapshots.items():
        status = {p["code"]: p["availability_status"] for p in snapshot["predictions"]}
        for e in snapshot["explanations"]:
            s = status.get(e["code"])
            if s not in FLAG_PREFERENCE or not is_complete(e):
                continue
            without = e["logit"] - contributions_by_concept(e)["Availability"]
            candidates.append((FLAG_PREFERENCE.index(s), -without, -gw, e["code"], s,
                               1 / (1 + math.exp(-without))))
    if not candidates:
        raise ValueError("no availability-flagged prediction with a complete explanation")
    best = min(candidates)
    return {"gameweek": -best[2], "code": best[3], "status": best[4], "p_without_availability": best[5]}


def _main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", default=CASE_STUDY_PATH)
    args = parser.parse_args()
    result = run(args.data_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    for name, s in result["specifications"].items():
        print("{0:<20} Brier {1:.5f}  Rotation {2:.5f}  slope {3:.3f}  ECE {4:.4f}  coefs {5}  max VIF {6:.1f}".format(
            name, s["logistic"]["overall"]["brier"], s["logistic"]["overall"]["rotation_brier"],
            s["logistic"]["overall"]["calibration_slope"], s["logistic"]["overall"]["ece"],
            s["n_coefficients"], s["max_vif"]))
    print("wrote", args.output)


if __name__ == "__main__":
    _main()
