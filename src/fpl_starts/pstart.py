"""Application-facing P(start) API: the frozen logistic model's predictions,
with their explanations, for dashboards and other read-only consumers.

Two sources, one result shape:

- `load_registered_predictions` -- the latest registered snapshot under
  predictions/ (written by `fpl-starts-logistic-predict`). This is the
  prospective evidence; it is only ever read here.
- `predict_logistic_p_start` -- the frozen model applied now to current
  inputs, via the same feature construction as `fpl-starts-logistic-predict`
  (`ml.predict.predict_round`). Nothing is fitted and nothing is written.

Both fail with a `PStartUnavailable` subclass when something required is
missing. There is no fallback to any other model.

This module lives outside `fpl_starts.ml` on purpose: the frozen model
records a hash of every source file in that package (`model_source_sha256`).
"""

import json
import os
import sqlite3
from dataclasses import dataclass

import pandas as pd

from . import config, explanation as grouped
from .ml import logistic, predict, spec

SOURCE_REGISTERED = "registered_snapshot"
SOURCE_LIVE = "live_frozen_model"

PLAYER_COLUMNS = [
    "code", "web_name", "team", "season", "gameweek", "p_start", "availability_status", "last_gw_role",
    "current_season_start_rate", "previous_season_start_rate", "cold_start", "logit", "team_code",
]
# Raw feature values surfaced as player columns, read from the explanation.
_EXPLANATION_COLUMNS = ["current_season_start_rate", "previous_season_start_rate"]


class PStartUnavailable(Exception):
    """A P(start) could not be produced or read."""


class ModelArtefactNotFound(PStartUnavailable, FileNotFoundError):
    pass


class InputDataUnavailable(PStartUnavailable):
    pass


class NoPredictionAvailable(PStartUnavailable, LookupError):
    pass


@dataclass
class PStartPredictions:
    """`players`: one row per player (PLAYER_COLUMNS). `contributions`: one
    row per (player, transformed feature) -- raw value, coefficient and
    log-odds contribution; intercept + a player's contributions = his logit.
    `metadata`: model id, source and timing of the predictions, and the
    nailed-on reference. `explanation`: one row per (player, group) compared
    with a nailed-on starter (see fpl_starts.explanation)."""
    players: pd.DataFrame
    contributions: pd.DataFrame
    metadata: dict
    explanation: pd.DataFrame

    def explain(self, code):
        """A player's grouped explanation, most-limiting group first."""
        return self.explanation[self.explanation["code"] == code].reset_index(drop=True)


def model_path(models_dir=config.MODELS_DIR):
    return os.path.join(logistic.model_dir(models_dir), "model.json")


def load_frozen_model(models_dir=config.MODELS_DIR):
    """The frozen `spec.MODEL_ID` model, loaded from its stored artefact."""
    path = model_path(models_dir)
    if not os.path.isfile(path):
        raise ModelArtefactNotFound("Logistic model artefact not found: {0}".format(path))
    with open(path) as f:
        stored = json.load(f)
    if stored.get("model_id") != spec.MODEL_ID:
        raise ModelArtefactNotFound("{0} holds model {1!r}, not {2}".format(path, stored.get("model_id"), spec.MODEL_ID))
    return logistic.FittedModel.from_dict(stored)


def _connect_read_only(db_path):
    if not os.path.isfile(db_path):
        raise InputDataUnavailable("Current availability data unavailable: no derived database at {0} -- "
                                   "run fpl-starts-derive first".format(db_path))
    return sqlite3.connect("file:{0}?mode=ro".format(os.path.abspath(db_path)), uri=True)


def _names(conn):
    players = pd.read_sql("SELECT code, web_name FROM players", conn).set_index("code")["web_name"]
    teams = pd.read_sql("SELECT code, name FROM teams", conn).set_index("code")["name"]
    return players, teams


def _build(records, explanations, players, teams, season, target_round, metadata, model):
    df = pd.DataFrame(records)
    raw = {exp["code"]: {f["raw_feature"]: f["raw_value"] for f in exp["features"]} for exp in explanations}
    for column in _EXPLANATION_COLUMNS:
        df[column] = pd.to_numeric(df["code"].map(lambda c: raw[c].get(column)), errors="coerce")
    df["season"] = season
    df["gameweek"] = int(target_round)
    df["web_name"] = df["code"].map(players)
    df["team"] = df["team_code"].map(teams)
    contributions = pd.DataFrame([
        dict(code=exp["code"], description=spec.DESCRIPTIONS.get(f["feature"]), **f)
        for exp in explanations for f in exp["features"]
    ])
    _, reference_logit = grouped.reference_values(model)
    metadata = dict(metadata, reference_p_start=grouped.sigmoid(reference_logit),
                    reference_description=grouped.NAILED_ON_DESCRIPTION)
    explained = grouped.grouped_explanation(contributions, df.set_index("code")["logit"].to_dict(), target_round, model)
    return PStartPredictions(df[PLAYER_COLUMNS], contributions, metadata, explained)


def load_registered_predictions(season, target_round, predictions_dir=config.PREDICTIONS_DIR,
                                db_path=config.DERIVED_DB_PATH, models_dir=config.MODELS_DIR):
    """The latest registered `spec.MODEL_ID` snapshot for (season,
    target_round), read-only, with player and team names from derived.db and
    the grouped explanation from the frozen model that made it."""
    season_dir = os.path.join(predictions_dir, season)
    candidates = []
    if os.path.isdir(season_dir):
        for name in sorted(os.listdir(season_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(season_dir, name)) as f:
                payload = json.load(f)
            if payload.get("model_version") == spec.MODEL_VERSION and payload["target_round"] == int(target_round):
                candidates.append(payload)
    if not candidates:
        raise NoPredictionAvailable("No prediction available for this gameweek: no registered {0} snapshot "
                                    "for {1} GW{2} under {3}".format(spec.MODEL_ID, season, target_round, season_dir))
    payload = max(candidates, key=lambda p: p["predicted_at"])
    model_id = payload.get("model", {}).get("model_id")
    if model_id != spec.MODEL_ID:
        raise NoPredictionAvailable("No prediction available for this gameweek: the {0} GW{1} snapshot was made by "
                                    "{2!r}, not {3}".format(season, target_round, model_id, spec.MODEL_ID))
    model = load_frozen_model(models_dir)
    if payload["model"].get("model_sha256") != logistic.file_sha256(model_path(models_dir)):
        raise NoPredictionAvailable("No prediction available for this gameweek: the {0} GW{1} snapshot was made by "
                                    "a different {2} model file than {3}".format(
                                        season, target_round, spec.MODEL_ID, model_path(models_dir)))

    conn = _connect_read_only(db_path)
    try:
        players, teams = _names(conn)
    finally:
        conn.close()
    metadata = {
        "model_id": model_id, "source": SOURCE_REGISTERED, "season": season, "gameweek": int(target_round),
        "predicted_at": payload["predicted_at"], "prediction_cutoff": payload.get("prediction_cutoff"),
        "deadline": payload.get("deadline"), "generated_after_deadline": payload.get("generated_after_deadline"),
        "model_sha256": payload["model"].get("model_sha256"), "intercept": payload["model"].get("intercept"),
    }
    return _build(payload["predictions"], payload["explanations"], players, teams, season, target_round, metadata,
                  model)


def predict_logistic_p_start(model, season, target_round, db_path=config.DERIVED_DB_PATH,
                             data_dir=config.DATA_DIR, raw_dir=config.RAW_DIR):
    """The frozen `model` (from `load_frozen_model`) applied to the current
    pre-cutoff inputs for (season, target_round). Never fits, never writes a
    snapshot."""
    conn = _connect_read_only(db_path)
    try:
        records, explanations, cutoff, deadline = predict.predict_round(conn, data_dir, raw_dir, model,
                                                                        season, target_round)
        players, teams = _names(conn)
    except FileNotFoundError as exc:  # includes ml.data.MissingLocalDataError
        raise InputDataUnavailable("Current availability data unavailable: {0}".format(exc)) from exc
    except ValueError as exc:
        raise NoPredictionAvailable("No prediction available for {0} GW{1}: {2}".format(
            season, target_round, exc)) from exc
    finally:
        conn.close()
    metadata = {
        "model_id": spec.MODEL_ID, "source": SOURCE_LIVE, "season": season, "gameweek": int(target_round),
        "predicted_at": None, "prediction_cutoff": cutoff.isoformat() + "Z", "deadline": deadline,
        "generated_after_deadline": None, "model_created_at": model.metadata.get("created_at"),
        "intercept": model.intercept,
    }
    return _build(records, explanations, players, teams, season, target_round, metadata, model)
