"""Fitting, persisting, predicting with and explaining the logistic model.

A fitted model is fully described by its preprocessing statistics, an
intercept and one coefficient per transformed column -- stored as plain JSON
(`model.json`), not a pickled estimator. Predictions and explanations are
computed from those stored numbers, so what is explained is exactly what
was predicted:

    logit   = intercept + sum_j coefficient_j * transformed_j
    p_start = 1 / (1 + exp(-logit))

Fitting only ever sees rows from `spec.TRAINING_SEASONS`; anything else --
in particular any prospective-season row -- is refused.
"""

import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import spec
from .preprocessing import Preprocessor, model_frame, sigmoid


class TemporalBoundaryError(ValueError):
    """Rows outside the permitted training seasons reached model fitting."""


def assert_training_rows(df):
    seasons = set(df["season"].unique())
    outside = seasons - set(spec.TRAINING_SEASONS)
    if outside:
        raise TemporalBoundaryError(
            "only {0} rows may be used for fitting or tuning; got {1}".format(
                list(spec.TRAINING_SEASONS), sorted(outside)))
    if len(df) == 0:
        raise ValueError("no training rows")


class FittedModel:
    def __init__(self, preprocessor, intercept, coefficients, C, metadata=None):
        self.preprocessor = preprocessor
        self.intercept = float(intercept)
        self.coefficients = pd.Series(coefficients, dtype=float)[spec.transformed_feature_names()]
        self.C = C
        self.metadata = metadata or {}

    # -- scoring -------------------------------------------------------------
    def transform(self, df):
        return self.preprocessor.transform(model_frame(df))

    def logit(self, df):
        X = self.transform(df)
        return self.intercept + X.values @ self.coefficients.values

    def predict_proba(self, df):
        return sigmoid(self.logit(df))

    def explain(self, df):
        """Per row: every transformed feature's raw value, transformed value,
        coefficient and log-odds contribution, plus intercept, logit and
        p_start. Contributions sum to the logit exactly (up to float error)."""
        raw = model_frame(df)
        X = self.transform(df)
        contributions = X * self.coefficients.values
        logit = self.intercept + contributions.sum(axis=1)
        out = []
        for i, idx in enumerate(df.index):
            features = []
            for name in spec.transformed_feature_names():
                source = name.split("__")[0]
                raw_value = raw.at[idx, source]
                if isinstance(raw_value, (np.floating, float)) and np.isnan(raw_value):
                    raw_value = None
                elif isinstance(raw_value, np.generic):
                    raw_value = raw_value.item()
                features.append({
                    "feature": name,
                    "raw_feature": source,
                    "raw_value": raw_value,
                    "transformed_value": float(X.at[idx, name]),
                    "coefficient": float(self.coefficients[name]),
                    "contribution": float(contributions.at[idx, name]),
                })
            out.append({"intercept": self.intercept, "logit": float(logit.iloc[i]),
                        "p_start": float(sigmoid(logit.iloc[i])), "features": features})
        return out

    # -- persistence ---------------------------------------------------------
    def to_dict(self):
        return {
            "model_id": spec.MODEL_ID,
            "model_version": spec.MODEL_VERSION,
            "raw_features": list(spec.RAW_FEATURES),
            "transformed_features": spec.transformed_feature_names(),
            "preprocessing": self.preprocessor.to_dict(),
            "missing_value_rule": spec.MISSING_VALUE_RULE,
            "intercept": self.intercept,
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "descriptions": spec.DESCRIPTIONS,
            "regularisation": {"penalty": "l2", "C": self.C},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d):
        if d["transformed_features"] != spec.transformed_feature_names():
            raise ValueError("stored model's feature order does not match this code's spec")
        return cls(Preprocessor.from_dict(d["preprocessing"]), d["intercept"], d["coefficients"],
                   d["regularisation"]["C"], d.get("metadata"))


def _estimator(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-8)


def fit(train, C, metadata=None):
    """Fit preprocessing and the L2 logistic regression on `train` only."""
    assert_training_rows(train)
    pre = Preprocessor().fit(model_frame(train))
    X = pre.transform(model_frame(train))
    est = _estimator(C).fit(X.values, train["y"].values)
    return FittedModel(pre, est.intercept_[0], dict(zip(X.columns, est.coef_[0])), C, metadata)


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def choose_C(train, validation, grid=spec.C_GRID):
    """Pick C by Brier on a strictly later validation window. Both windows
    must be training-season rows."""
    assert_training_rows(train)
    assert_training_rows(validation)
    scores = {c: brier(fit(train, c).predict_proba(validation), validation["y"]) for c in grid}
    best = min(grid, key=lambda c: (scores[c], c))
    return best, scores


# --- Files -----------------------------------------------------------------------

def model_dir(models_dir, model_id=spec.MODEL_ID):
    return os.path.join(models_dir, spec.MODEL_VERSION, model_id)


def save(model, directory):
    """Write-once: a fitted model is frozen, so an existing one is never
    overwritten -- a changed model is a new MODEL_ID."""
    path = os.path.join(directory, "model.json")
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite frozen model: {0}".format(path))
    os.makedirs(directory, exist_ok=True)
    body = json.dumps(model.to_dict(), indent=2, sort_keys=True, default=str)
    with open(path, "w") as f:
        f.write(body)
    return path


def load(directory):
    path = os.path.join(directory, "model.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "no fitted model at {0} -- run `fpl-starts-logistic-train` first".format(path))
    with open(path) as f:
        return FittedModel.from_dict(json.load(f))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
