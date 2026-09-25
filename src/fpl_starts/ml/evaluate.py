"""Historical walk-forward evaluation on the development seasons.

Folds are season-granular and strictly chronological; preprocessing and C
are learned inside each fold's training window only:

    test 2024-25: train 2023-24 (C chosen on its last 25% of gameweeks)
    test 2025-26: train 2023-24, C chosen on 2024-25, refit on both

Both test seasons were also used while choosing the feature specification,
so these are development results, not an untouched test. The prospective
evidence is 2026-27, scored by the frozen model (see predict.py).

Benchmark: naive P(start | started his previous gameweek), a two-number
table fitted on the same training window -- the simplest sensible P(start).

Usage: fpl-starts-logistic-evaluate [--data-dir data] [--output PATH]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config
from . import logistic, panel as mlpanel, spec

FOLDS = [
    ("2024-25", ["2023-24"]),
    ("2025-26", ["2023-24", "2024-25"]),
]
STRATA = ["Core", "Rotation", "Marginal", "Deep"]


def split_train_validation(df, train_seasons):
    """(train, validation): the last training season validates; a single
    training season is split at 75% of its gameweeks."""
    if len(train_seasons) == 1:
        season = df[df["season"] == train_seasons[0]]
        rounds = sorted(season["round"].unique())
        cut = rounds[int(len(rounds) * 0.75)]
        return season[season["round"] < cut], season[season["round"] >= cut]
    return (df[df["season"].isin(train_seasons[:-1])], df[df["season"] == train_seasons[-1]])


def naive_baseline(train, test):
    """P(start | started previous gameweek at this club), fitted on `train`;
    rows with no previous gameweek at the club form their own cell."""
    key_train = train["prev_started"].fillna(-1)
    rates = train.groupby(key_train)["y"].mean()
    return test["prev_started"].fillna(-1).map(rates).fillna(train["y"].mean()).values


def calibration(p, y, n_bins=10):
    """Logistic recalibration slope/intercept (1, 0 = perfect) and expected
    calibration error over equal-width probability bins."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    bins = pd.cut(p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), mean_observed=("y", "mean"), n=("y", "size"))
    ece = float((table["n"] * (table["mean_predicted"] - table["mean_observed"]).abs()).sum() / table["n"].sum())
    return {"slope": float(fit.coef_[0][0]), "intercept": float(fit.intercept_[0]), "ece": ece,
            "bins": [{"mean_predicted": float(r.mean_predicted), "mean_observed": float(r.mean_observed),
                      "n": int(r.n)} for r in table.itertuples()]}


def metrics(p, y, strata):
    out = {"n": int(len(y)), "brier": logistic.brier(p, y), "by_stratum": {}}
    for s in STRATA:
        m = np.asarray(strata) == s
        if m.any():
            out["by_stratum"][s] = {"n": int(m.sum()), "brier": logistic.brier(np.asarray(p)[m], np.asarray(y)[m])}
    out["calibration"] = calibration(p, y)
    return out


def run(panel):
    df = panel[panel["season"].isin(spec.TRAINING_SEASONS)].copy()
    folds, preds = [], []
    for test_season, train_seasons in FOLDS:
        train, validation = split_train_validation(df, train_seasons)
        C, scores = logistic.choose_C(train, validation)
        model = logistic.fit(pd.concat([train, validation]), C)
        test = df[df["season"] == test_season]
        fold_pred = test[["season", "round", "code", "y", "stratum"]].copy()
        fold_pred["logistic"] = model.predict_proba(test)
        fold_pred["naive"] = naive_baseline(pd.concat([train, validation]), test)
        preds.append(fold_pred)
        folds.append({"test_season": test_season, "train_seasons": train_seasons, "C": C,
                      "validation_brier_by_C": {str(k): v for k, v in scores.items()},
                      "n_train": int(len(train) + len(validation)), "n_test": int(len(test)),
                      "intercept": model.intercept,
                      "coefficients": {k: float(v) for k, v in model.coefficients.items()}})
    pred = pd.concat(preds, ignore_index=True)
    result = {"note": "development results: 2024-25 and 2025-26 informed the feature "
                      "specification, so neither is an untouched test set",
              "folds": folds, "by_test_season": {}, "overall": {}}
    for label, rows in [("overall", pred)] + [(s, pred[pred["season"] == s]) for s, _ in FOLDS]:
        block = {name: metrics(rows[name].values, rows["y"].values, rows["stratum"].values)
                 for name in ["naive", "logistic"]}
        block["relative_brier_improvement"] = 1 - block["logistic"]["brier"] / block["naive"]["brier"]
        rot = block["logistic"]["by_stratum"].get("Rotation"), block["naive"]["by_stratum"].get("Rotation")
        if all(rot):
            block["relative_rotation_brier_improvement"] = 1 - rot[0]["brier"] / rot[1]["brier"]
        if label == "overall":
            result["overall"] = block
        else:
            result["by_test_season"][label] = block
    return result, pred


def format_report(result):
    lines = ["Historical walk-forward (development seasons; not an untouched test)", ""]
    header = "{0:<10} {1:<9} {2:>8} {3:>8} {4:>8} {5:>8} {6:>8} {7:>7} {8:>7}".format(
        "window", "model", "Brier", "Core", "Rotation", "Marginal", "Deep", "slope", "ECE")
    lines.append(header)
    blocks = [("overall", result["overall"])] + list(result["by_test_season"].items())
    for label, block in blocks:
        for name in ["naive", "logistic"]:
            m = block[name]
            s = m["by_stratum"]
            lines.append("{0:<10} {1:<9} {2:>8.4f} {3:>8.4f} {4:>8.4f} {5:>8.4f} {6:>8.4f} {7:>7.3f} {8:>7.4f}".format(
                label, name, m["brier"], s["Core"]["brier"], s["Rotation"]["brier"],
                s["Marginal"]["brier"], s["Deep"]["brier"], m["calibration"]["slope"], m["calibration"]["ece"]))
        lines.append("{0:<10} relative Brier improvement: {1:.1%} overall, {2:.1%} Rotation".format(
            label, block["relative_brier_improvement"], block["relative_rotation_brier_improvement"]))
    lines.append("")
    lines.append("C chosen per fold: " + ", ".join("{0}: {1}".format(f["test_season"], f["C"]) for f in result["folds"]))
    return "\n".join(lines)


def _main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", default=os.path.join(config.MODELS_DIR, spec.MODEL_VERSION,
                                                          "historical_evaluation.json"))
    args = parser.parse_args()
    panel = mlpanel.build_historical_panel(args.data_dir)
    result, _ = run(panel)
    print(format_report(result))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", args.output)


if __name__ == "__main__":
    _main()
