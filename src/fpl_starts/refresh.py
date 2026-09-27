"""Refresh FPL availability on request, and re-forecast the upcoming gameweek.

One refresh, for everyone using this data:

1. Only before the upcoming gameweek's deadline -- the gameweek after the
   last one with results in derived.db. After its deadline that gameweek's
   forecast is final, so nothing is fetched.
2. At most once per `cooldown` (default 30 minutes), judged on the archive's
   latest bootstrap-static capture -- however many people ask.
3. One at a time: a lock file beside the database.

Then: fetch and archive bootstrap-static (write-once, exactly as the
scheduled archive does), rebuild derived.db atomically, re-run the frozen
model for the upcoming gameweek, and register a new forecast snapshot only
if any player's chance of starting changed -- rebuilding once more to load
it. The fitted model is never touched and nothing is refitted.

Usage: fpl-starts-refresh [--base-dir raw] [--db-path db/derived.db] ...
"""

import argparse
import fcntl
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import api, archiver, config, derived
from .ml import logistic, predict, spec
from .ml.data import MissingLocalDataError

COOLDOWN = timedelta(minutes=30)

REFRESHED = "refreshed"
RECENT = "recent"
DEADLINE_PASSED = "deadline_passed"
SEASON_OVER = "season_over"


@dataclass
class RefreshResult:
    status: str               # REFRESHED, RECENT, DEADLINE_PASSED or SEASON_OVER
    message: str              # plain English, for the user
    gameweek: int             # the upcoming gameweek
    data_as_of: str           # latest capture ("YYYYMMDDTHHMMSSZ") after this call
    forecast_updated: bool = False
    snapshot_path: str = None


def _parse(timestamp):
    """A capture's fetched_at or an FPL deadline_time, as aware UTC."""
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(timestamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError("unrecognised timestamp: {0!r}".format(timestamp))


def _describe(when):
    return when.strftime("%d %b %H:%M UTC")


def _last_completed_gameweek(db_path, season):
    conn = sqlite3.connect("file:{0}?mode=ro".format(os.path.abspath(db_path)), uri=True)
    try:
        (gw,) = conn.execute("SELECT MAX(round) FROM player_gameweek_stats WHERE season = ?", (season,)).fetchone()
    finally:
        conn.close()
    return gw or 0


def _latest_capture(base_dir, season):
    return derived.latest_by(derived.ok_entries(base_dir, season, "bootstrap-static"), lambda e: e["fetched_at"])


def _deadline(payload, gameweek):
    for event in payload.get("events") or []:
        if event.get("id") == gameweek and event.get("deadline_time"):
            return _parse(event["deadline_time"])
    return None


def _latest_registered(predictions_dir, season, gameweek):
    """{code: p_start} from the latest registered forecast for `gameweek`, or None."""
    season_dir = os.path.join(predictions_dir, season)
    latest = None
    if os.path.isdir(season_dir):
        for name in os.listdir(season_dir):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(season_dir, name)) as f:
                payload = json.load(f)
            if payload.get("model_version") == spec.MODEL_VERSION and payload.get("target_round") == gameweek:
                if latest is None or payload["predicted_at"] > latest["predicted_at"]:
                    latest = payload
    return None if latest is None else {r["code"]: r["p_start"] for r in latest["predictions"]}


def _forecast(base_dir, db_path, predictions_dir, models_dir, data_dir, season, gameweek, clock):
    """Re-run the frozen model; register a snapshot only if a chance changed.
    Returns (updated, snapshot_path, reason-if-not)."""
    directory = logistic.model_dir(models_dir)
    model = logistic.load(directory)
    conn = sqlite3.connect("file:{0}?mode=ro".format(os.path.abspath(db_path)), uri=True)
    try:
        records, detail, cutoff, deadline = predict.predict_round(conn, data_dir, base_dir, model, season, gameweek)
    except (ValueError, MissingLocalDataError) as exc:
        return False, None, str(exc)
    finally:
        conn.close()
    previous = _latest_registered(predictions_dir, season, gameweek)
    current = {r["code"]: r["p_start"] for r in records}
    if previous is not None and previous.keys() == current.keys() and all(
            abs(previous[c] - current[c]) < 1e-12 for c in current):
        return False, None, "no player's chance of starting changed"
    path = predict.write_snapshot(records, detail, model, os.path.join(directory, "model.json"), season, gameweek,
                                  cutoff, deadline, base_dir=predictions_dir, clock=clock)
    return True, path, None


def refresh(base_dir=archiver.RAW_DIR, db_path=config.DERIVED_DB_PATH, predictions_dir=config.PREDICTIONS_DIR,
            models_dir=config.MODELS_DIR, data_dir=config.DATA_DIR, season=spec.PROSPECTIVE_SEASON,
            http_get=None, clock=archiver.utcnow, cooldown=COOLDOWN):
    """Refresh availability and the upcoming gameweek's forecast, if allowed
    now -- see the module docstring. Returns a RefreshResult."""
    lock_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(lock_dir, exist_ok=True)
    with open(os.path.join(lock_dir, ".refresh.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # a second caller waits, then usually finds the data RECENT
        try:
            return _refresh(base_dir, db_path, predictions_dir, models_dir, data_dir, season, http_get, clock,
                            cooldown)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _refresh(base_dir, db_path, predictions_dir, models_dir, data_dir, season, http_get, clock, cooldown):
    now = clock()
    gameweek = _last_completed_gameweek(db_path, season) + 1
    latest = _latest_capture(base_dir, season)
    as_of = latest["fetched_at"] if latest else None
    if gameweek > 38:
        return RefreshResult(SEASON_OVER, "The season is over -- there's no upcoming gameweek to forecast.",
                             gameweek, as_of)
    if latest is not None:
        deadline = _deadline(derived.read_gz_json(archiver.entry_file(base_dir, latest)), gameweek)
        if deadline is not None and now >= deadline:
            return RefreshResult(DEADLINE_PASSED, "The gameweek {0} deadline ({1}) has passed, so its forecast is "
                                 "final and our FPL data can't be refreshed until gameweek {0} has been played."
                                 .format(gameweek, _describe(deadline)), gameweek, as_of)
        fetched = _parse(latest["fetched_at"])
        if now - fetched < cooldown:
            return RefreshResult(RECENT, "Our FPL data was refreshed at {0}, which is recent enough -- FPL news "
                                 "rarely changes faster than that.".format(_describe(fetched)), gameweek, as_of)

    if http_get is None:
        http_get = archiver.make_http_get(api.new_session())
    entry = archiver.archive_snapshot("bootstrap-static", http_get, season=season, base_dir=base_dir, clock=clock)
    derived.rebuild_and_swap(base_dir=base_dir, db_path=db_path, predictions_dir=predictions_dir)
    updated, path, reason = _forecast(base_dir, db_path, predictions_dir, models_dir, data_dir, season, gameweek,
                                      clock)
    if updated:
        derived.rebuild_and_swap(base_dir=base_dir, db_path=db_path, predictions_dir=predictions_dir)
        message = "Refreshed our FPL data and updated the gameweek {0} forecast.".format(gameweek)
    else:
        message = "Refreshed our FPL data; the gameweek {0} forecast is unchanged ({1}).".format(gameweek, reason)
    return RefreshResult(REFRESHED, message, gameweek, entry["fetched_at"], updated, path)


def _main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", default=archiver.RAW_DIR)
    parser.add_argument("--db-path", default=config.DERIVED_DB_PATH)
    parser.add_argument("--predictions-dir", default=config.PREDICTIONS_DIR)
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--cooldown-minutes", type=float, default=COOLDOWN.total_seconds() / 60)
    args = parser.parse_args()
    result = refresh(args.base_dir, args.db_path, args.predictions_dir, args.models_dir, args.data_dir,
                     cooldown=timedelta(minutes=args.cooldown_minutes))
    print(result.message)
    if result.snapshot_path:
        print("forecast snapshot:", result.snapshot_path)


if __name__ == "__main__":
    _main()
