"""Read-only data access for the dashboard and agent: both repos' derived.db
files, and the live (public, unauthenticated) FPL manager-team API.

Deliberately never writes to either derived.db -- this project's derived
layer is disposable/rebuilt-from-archive by design (see fpl-starts's
docs/build_spec_p_starts.md), and a dashboard has no business being a
second writer to it. Everything here is a plain read or an external GET.
"""

import os
import sqlite3

import pandas as pd
import requests

FPL_API_BASE = "https://fantasy.premierleague.com/api"
_USER_AGENT = "Mozilla/5.0 (compatible; fpl-dashboard/0.1)"

# This repo's own derived.db (base + agent arms) and the private news repo's
# (news arms) -- same pair gameweek_report.py (../../fpl/src/gameweek_report.py)
# reads, see that script's --db-path/--fpl-starts-dir defaults. Overridable
# via env var for anyone running the dashboard from a different working
# directory, or without the private repo checked out at all (the fpl-starts
# side works standalone; FPL_NEWS_DB_PATH is optional).
FPL_STARTS_DB_PATH = os.environ.get("FPL_DASHBOARD_FPL_STARTS_DB", os.path.join("..", "db", "derived.db"))
FPL_NEWS_DB_PATH = os.environ.get(
    "FPL_DASHBOARD_NEWS_DB", os.path.join("..", "..", "fpl", "db", "derived.db")
)


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


def _connect(db_path):
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            "no database at {0} -- run the dashboard from fpl-starts/dashboard/, "
            "or set FPL_DASHBOARD_FPL_STARTS_DB / FPL_DASHBOARD_NEWS_DB".format(db_path)
        )
    return sqlite3.connect(db_path)


def load_predictions(db_path, season, target_round, codes=None):
    """code, web_name, p_start, cold_start, method, model_version for every
    archived prediction in `target_round` -- optionally filtered to
    `codes` (e.g. one manager's 15-player squad).
    """
    conn = _connect(db_path)
    sql = (
        "SELECT pr.code, p.web_name, pr.p_start, pr.cold_start, pr.method, "
        "pr.model_version FROM predictions pr "
        "LEFT JOIN players p ON p.code = pr.code "
        "WHERE pr.season = ? AND pr.target_round = ?"
    )
    params = [season, target_round]
    if codes:
        sql += " AND pr.code IN ({0})".format(", ".join(["?"] * len(codes)))
        params += list(codes)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def load_squad_predictions(team_id, event, season, target_round):
    """One manager's 15 picks joined to every arm's p_start for that round,
    across both repos' derived.db -- the "give me my team ID" feature.
    Columns: code, web_name, position, multiplier, is_captain,
    is_vice_captain, plus one p_start column per model_version found.
    """
    bootstrap = fetch_bootstrap()
    id_to_code = {e["id"]: e["code"] for e in bootstrap["elements"]}

    picks_payload = fetch_team_picks(team_id, event)
    picks = pd.DataFrame(picks_payload["picks"])
    picks["code"] = picks["element"].map(id_to_code)
    codes = picks["code"].dropna().astype(int).tolist()

    frames = []
    for db_path in (FPL_STARTS_DB_PATH, FPL_NEWS_DB_PATH):
        if os.path.isfile(db_path):
            frames.append(load_predictions(db_path, season, target_round, codes=codes))
    if not frames:
        raise FileNotFoundError("neither derived.db was found -- check FPL_DASHBOARD_* env vars")
    predictions = pd.concat(frames, ignore_index=True)

    wide = predictions.pivot_table(
        index=["code", "web_name"], columns="model_version", values="p_start", aggfunc="first"
    ).reset_index()

    merged = picks.merge(wide, on="code", how="left")
    return merged[["code", "web_name", "position", "multiplier", "is_captain",
                    "is_vice_captain"] + [c for c in wide.columns if c not in ("code", "web_name")]]


def load_gameweek_comparison(season, prior_season, target_round):
    """Stratified Brier/accuracy per model_version, across both repos'
    derived.db, reusing fpl_starts.scoring directly (no reimplementation --
    same building block gameweek_report.py uses, minus the deadline/
    quarantine bookkeeping that only matters for archiving new predictions,
    not for reading already-clean ones back).
    """
    from fpl_starts.scoring import compare_models, list_model_versions

    reports = {}
    for label, db_path in (("fpl-starts", FPL_STARTS_DB_PATH), ("fpl", FPL_NEWS_DB_PATH)):
        if not os.path.isfile(db_path):
            continue
        conn = _connect(db_path)
        versions = list_model_versions(conn, season, target_round)
        if versions:
            reports[label] = compare_models(conn, season, prior_season, target_round, versions)
        conn.close()
    return reports
