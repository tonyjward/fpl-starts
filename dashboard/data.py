"""Read-only data access for the dashboard and agent: the frozen logistic
P(start) model (logistic_availability_v1) through `fpl_starts.pstart`, this
repo's derived.db, and the live (public, unauthenticated) FPL API.

P(start) comes from one of two places, both via `fpl_starts.pstart` -- the
dashboard never builds features or applies coefficients itself:

- registered snapshots under predictions/ (the prospective evidence, read
  only -- opening the dashboard never writes or replaces one);
- the frozen model applied live to current inputs (nothing fitted, nothing
  written).

Deliberately never writes to derived.db -- this project's derived layer is
disposable/rebuilt-from-archive by design, and a dashboard has no business
being a second writer to it. Everything here is a plain read or an external
GET.
"""

import os
import sqlite3

import pandas as pd
import requests

from fpl_starts import config, pstart
from fpl_starts.ml import spec

FPL_API_BASE = "https://fantasy.premierleague.com/api"
_USER_AGENT = "Mozilla/5.0 (compatible; fpl-dashboard/0.1)"

# This repo's local runtime data, resolved from the repo root so the app
# works from any working directory. FPL_DASHBOARD_FPL_STARTS_DB overrides the
# database path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FPL_STARTS_DB_PATH = os.environ.get("FPL_DASHBOARD_FPL_STARTS_DB",
                                    os.path.join(REPO_ROOT, config.DERIVED_DB_PATH))
MODELS_DIR = os.path.join(REPO_ROOT, config.MODELS_DIR)
PREDICTIONS_DIR = os.path.join(REPO_ROOT, config.PREDICTIONS_DIR)
DATA_DIR = os.path.join(REPO_ROOT, config.DATA_DIR)
RAW_DIR = os.path.join(REPO_ROOT, config.RAW_DIR)

SOURCE_REGISTERED = pstart.SOURCE_REGISTERED
SOURCE_LIVE = pstart.SOURCE_LIVE


def _get_json(path):
    resp = requests.get("{0}/{1}".format(FPL_API_BASE, path),
                         headers={"User-Agent": _USER_AGENT}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap():
    """Live bootstrap-static -- id/code/name/team mapping. Not cached to
    disk (this is a dashboard, not the write-once archive); Streamlit's own
    @st.cache_data (applied by callers, not here, to keep this module
    framework-agnostic) is the right place for any caching.
    """
    return _get_json("bootstrap-static/")


def fetch_team_picks(team_id, event):
    """One manager's squad for `event`: 15 picks (element id, position,
    multiplier, is_captain, is_vice_captain), plus entry_history (points,
    rank, etc.) and automatic_subs. Public, unauthenticated endpoint --
    raises requests.HTTPError (404) for an invalid team_id or an event
    that hasn't happened for that manager yet.
    """
    return _get_json("entry/{0}/event/{1}/picks/".format(team_id, event))


def fetch_team_summary(team_id):
    """Manager name and overall team name -- for labelling, not analysis."""
    return _get_json("entry/{0}/".format(team_id))


def load_frozen_model():
    """The frozen logistic model -- load once per process (st.cache_resource)."""
    return pstart.load_frozen_model(MODELS_DIR)


def load_gameweek_predictions(season, target_round, source=SOURCE_REGISTERED, model=None):
    """`pstart.PStartPredictions` for one gameweek: the registered snapshot,
    or (source=SOURCE_LIVE) the frozen `model` applied to current inputs."""
    if source == SOURCE_REGISTERED:
        return pstart.load_registered_predictions(season, target_round, PREDICTIONS_DIR, FPL_STARTS_DB_PATH,
                                                  MODELS_DIR)
    if source == SOURCE_LIVE:
        if model is None:
            model = load_frozen_model()
        return pstart.predict_logistic_p_start(model, season, target_round, FPL_STARTS_DB_PATH, DATA_DIR, RAW_DIR)
    raise ValueError("unknown P(start) source: {0!r}".format(source))


def load_squad_predictions(team_id, event, season, target_round, predictions=None):
    """One manager's 15 picks joined to the logistic P(start) for that
    round -- the "give me my team ID" feature. `predictions` defaults to
    the registered snapshot for `target_round`.
    """
    bootstrap = fetch_bootstrap()
    id_to_code = {e["id"]: e["code"] for e in bootstrap["elements"]}

    picks_payload = fetch_team_picks(team_id, event)
    picks = pd.DataFrame(picks_payload["picks"])
    picks["code"] = picks["element"].map(id_to_code)

    if predictions is None:
        predictions = load_gameweek_predictions(season, target_round)
    columns = ["code", "web_name", "team", "p_start", "availability_status", "last_gw_role",
               "current_season_start_rate", "previous_season_start_rate"]
    merged = picks.merge(predictions.players[columns], on="code", how="left")
    return merged[["code", "web_name", "team", "position", "multiplier", "is_captain", "is_vice_captain",
                   "p_start", "availability_status", "last_gw_role",
                   "current_season_start_rate", "previous_season_start_rate"]]


def load_gameweek_comparison(season, prior_season, target_round, fetch=None):
    """Stratified Brier/accuracy of the logistic model's registered
    predictions against the baselines, via fpl_starts.scoring directly.
    {} when the round hasn't been predicted and played yet."""
    from fpl_starts.scoring import ScoringError, score_gameweek

    if not os.path.isfile(FPL_STARTS_DB_PATH):
        raise FileNotFoundError("no database at {0} -- run fpl-starts-derive, or set "
                                "FPL_DASHBOARD_FPL_STARTS_DB".format(FPL_STARTS_DB_PATH))
    conn = sqlite3.connect(FPL_STARTS_DB_PATH)
    try:
        report = score_gameweek(conn, season, prior_season, target_round,
                                model_version=spec.MODEL_VERSION, fetch=fetch)
    except ScoringError:
        return {}
    finally:
        conn.close()
    return {spec.MODEL_ID: report}
