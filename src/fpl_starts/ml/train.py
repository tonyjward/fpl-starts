"""Fit the ONE frozen logistic P(start) model.

Training rows: every fixture of 2023-24, 2024-25 and 2025-26 (2022-23 only
provides history). C is chosen by training on 2023-24 + 2024-25 and
validating on 2025-26, then the model is refitted on all three seasons.
Nothing from 2026-27 is loaded.

The result is written once to models/logistic_availability/<model id>/ and
never overwritten: 2026-27 is scored by this exact model all season. An
improved model is a new model id with its own prospective record.

Usage: fpl-starts-logistic-train [--data-dir data] [--models-dir models]
"""

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import sklearn

from .. import config
from . import data as mldata, evaluate, logistic, panel as mlpanel, spec


def input_fingerprint(data_dir):
    files = [mldata.HISTORICAL_AVAILABILITY, mldata.DEADLINES]
    for season in spec.HISTORICAL_SEASONS:
        for name in (os.path.join("gws", "merged_gw.csv"), "players_raw.csv", "teams.csv"):
            files.append(os.path.join("vaastav", season, name))
    return {f: logistic.file_sha256(mldata.require(data_dir, f)) for f in sorted(files)}


def matrix_fingerprint(model, rows):
    """sha256 of the exact transformed training matrix and targets."""
    X = model.transform(rows)
    h = pd.util.hash_pandas_object(pd.concat([X, rows["y"]], axis=1), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return None


def source_fingerprint():
    """sha256 over this package's model source files -- identifies the exact
    code that fitted a model even when it was not yet committed."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(here) if f.endswith(".py")):
        with open(os.path.join(here, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


def train(data_dir, models_dir):
    directory = logistic.model_dir(models_dir)
    if os.path.exists(os.path.join(directory, "model.json")):
        raise FileExistsError("{0} is already fitted and frozen: {1}".format(spec.MODEL_ID, directory))

    panel = mlpanel.build_historical_panel(data_dir)
    rows = panel[panel["season"].isin(spec.TRAINING_SEASONS)].copy()

    tune_train = rows[rows["season"].isin(spec.TRAINING_SEASONS[:-1])]
    tune_validation = rows[rows["season"] == spec.TRAINING_SEASONS[-1]]
    C, scores = logistic.choose_C(tune_train, tune_validation)

    evaluation, _ = evaluate.run(panel)

    metadata = {
        "created_at": logistic.utcnow_iso(),
        "training_seasons": list(spec.TRAINING_SEASONS),
        "context_only_seasons": list(spec.CONTEXT_ONLY_SEASONS),
        "prospective_season": spec.PROSPECTIVE_SEASON,
        "prospective_rule": "frozen: never refitted, retuned or recalibrated on prospective-season data",
        "target": "y = 1 if the player started the fixture",
        "unit_of_observation": "player x fixture; double-gameweek fixtures share one feature vector",
        "prediction_cutoff": "deadline - {0}h".format(spec.CUTOFF_HOURS_BEFORE_DEADLINE),
        "training_rows": int(len(rows)),
        "training_rows_by_season": {k: int(v) for k, v in rows.groupby("season").size().items()},
        "training_start_rate": float(rows["y"].mean()),
        "C_selection": {"train_seasons": list(spec.TRAINING_SEASONS[:-1]),
                        "validation_season": spec.TRAINING_SEASONS[-1],
                        "criterion": "Brier", "validation_brier_by_C": {str(k): v for k, v in scores.items()},
                        "chosen_C": C},
        "input_sha256": input_fingerprint(data_dir),
        "fpl_starts_git_commit": _git("rev-parse", "HEAD"),
        "fpl_starts_uncommitted_changes": bool(_git("status", "--porcelain")),
        "model_source_sha256": source_fingerprint(),
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__},
        "historical_evaluation_summary": {
            "note": evaluation["note"],
            "brier": {"naive": evaluation["overall"]["naive"]["brier"],
                      "logistic": evaluation["overall"]["logistic"]["brier"]},
            "relative_brier_improvement": evaluation["overall"]["relative_brier_improvement"],
            "relative_rotation_brier_improvement": evaluation["overall"]["relative_rotation_brier_improvement"],
        },
    }
    model = logistic.fit(rows, C, metadata)
    model.metadata["training_matrix_sha256"] = matrix_fingerprint(model, rows)
    path = logistic.save(model, directory)
    with open(os.path.join(directory, "historical_evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    return model, path, evaluation


def _main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    args = parser.parse_args()
    model, path, evaluation = train(args.data_dir, args.models_dir)
    print(evaluate.format_report(evaluation))
    print("\nfrozen model written to", path)
    print("C = {0}, intercept = {1:+.3f}".format(model.C, model.intercept))
    for name, coef in model.coefficients.items():
        print("  {0:<36} {1:+.3f}".format(name, coef))


if __name__ == "__main__":
    _main()
