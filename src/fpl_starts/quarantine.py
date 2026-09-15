"""Move post-deadline prediction snapshots out of the way.

`derived.py` treats the *latest* archived snapshot per (season, round,
model_version) as the prediction for that round -- the right rule for the
normal case (a mid-gameweek rerun that supersedes an earlier one, both
made before the deadline). It breaks when a snapshot gets regenerated
*after* the round's deadline has passed, e.g. testing a code change
against a fixture that's already been decided: "latest" then silently
promotes hindsight-contaminated output over the real, once-real blind
prediction, and nothing about derived.py's own rebuild can tell the
difference -- both are just a JSON file with a timestamp.

This never deletes anything: it moves offending files from
`{predictions_dir}/{season}/` to `{predictions_dir}/post_deadline/{season}/`,
a sibling directory derived.py's flat, non-recursive `os.listdir` never
sees, so the real pre-deadline snapshot (if one was archived) becomes
"latest" again on the next rebuild. A file moved here by mistake, or
wanted for a deliberate one-off retrospective look (see
docs/gameweek-summary.md's Leeds v Newcastle case), is still right there
under a normal path -- nothing about a house move is a diagnosis.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import json
import os
from datetime import datetime

from . import derived
from .config import PREDICTIONS_DIR, RAW_DIR

_DEADLINE_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _round_deadlines(raw_dir, season):
    """{round: deadline_time (UTC datetime)} from the most recently
    archived bootstrap-static snapshot -- its events[] list covers the
    whole season, so this works for a past round even from today's
    snapshot. {} if no snapshot has been archived at all yet.
    """
    payload = derived.latest_bootstrap_payload(raw_dir, season)
    if payload is None:
        return {}
    deadlines = {}
    for event in payload.get("events") or []:
        gw, deadline = event.get("id"), event.get("deadline_time")
        if gw is not None and deadline:
            deadlines[gw] = datetime.strptime(deadline, _DEADLINE_FMT)
    return deadlines


def quarantine_post_deadline(predictions_dir, raw_dir, season, dry_run=False):
    """Move every snapshot in `{predictions_dir}/{season}/` whose
    `predicted_at` postdates its own `target_round`'s deadline into
    `{predictions_dir}/post_deadline/{season}/`.

    Returns {"moved": [...], "kept": [...], "unknown": [...]} (each a list
    of filenames) -- "unknown" is a round with no deadline on record (an
    unarchived bootstrap-static, or a round past the end of the season
    list), left untouched rather than guessed at.
    """
    season_dir = os.path.join(predictions_dir, season)
    result = {"moved": [], "kept": [], "unknown": []}
    if not os.path.isdir(season_dir):
        return result

    deadlines = _round_deadlines(raw_dir, season)
    dest_dir = os.path.join(predictions_dir, "post_deadline", season)

    for name in sorted(os.listdir(season_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(season_dir, name)
        with open(path) as f:
            payload = json.load(f)
        target_round = payload["target_round"]
        predicted_at = derived.parse_fetched_at(payload["predicted_at"])
        deadline = deadlines.get(target_round)

        if deadline is None:
            result["unknown"].append(name)
        elif predicted_at > deadline:
            result["moved"].append(name)
            if not dry_run:
                if not os.path.isdir(dest_dir):
                    os.makedirs(dest_dir)
                os.rename(path, os.path.join(dest_dir, name))
        else:
            result["kept"].append(name)

    return result


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Move prediction snapshots whose predicted_at postdates "
                    "their round's deadline out of the season directory, so "
                    "derived.py's rebuild no longer treats them as 'latest'."
    )
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--season", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would move without moving anything.")
    args = parser.parse_args()

    result = quarantine_post_deadline(
        args.predictions_dir, args.raw_dir, args.season, dry_run=args.dry_run
    )
    verb = "would move" if args.dry_run else "moved"
    for name in result["moved"]:
        print("{0}: {1}".format(verb, name))
    for name in result["unknown"]:
        print("unknown deadline, left alone: {0}".format(name))
    print("{0}: {1}, kept: {2}, unknown: {3}".format(
        verb, len(result["moved"]), len(result["kept"]), len(result["unknown"])
    ))


if __name__ == "__main__":
    _main()
