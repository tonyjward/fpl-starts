"""Per-domain accuracy monitoring for the agent's evidence sources.

Tracks which specific base URL/domain a verified classification came from,
and (once real outcomes exist) whether that domain's directional claims
(confirmed_out/confirmed_starting -- the two categories that assert a
specific, checkable outcome) turned out right. This is monitoring
infrastructure only: nothing here feeds back into predict.py's blend yet.
categories.category_to_p_start weights every surviving classification
equally regardless of domain, same as it does today -- the private repo's
own per-source-tier weighting stayed uniform for the same reason (nowhere
near enough claims per tier to weight on, see their future_refinements.md).
Per-domain will need even more volume than per-tier before there's a real
signal to act on; this module is what makes that signal visible once it
exists, not an assumption that it exists yet.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import json
import os
from urllib.parse import urlsplit

import pandas as pd

# The two categories that assert a specific, checkable outcome -- same
# scope the private repo's own directional-accuracy scoring uses
# (_DIRECTIONAL_CLAIM_CATEGORIES in their news_scoring.py).
# rotation_risk/returning_from_injury don't assert a specific outcome, so
# there's no correct/incorrect call to score for them the same way.
DIRECTIONAL_CATEGORIES = {"confirmed_out": 0, "confirmed_starting": 1}


def _domain_of(url):
    """Bare domain from a URL -- lowercased, a leading "www." stripped,
    path/query/scheme ignored. "" for an unparseable or empty url.
    """
    if not url:
        return ""
    netloc = urlsplit(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[len("www."):]
    return netloc


def rebuild_evidence_table(conn, predictions_dir, season,
                            model_version="refined_availability_agent_news"):
    """Rebuild `agent_evidence` from scratch by replaying the raw
    `predictions/{season}/*.json` snapshot files -- same "derived layer is
    disposable, rebuilt freely" discipline as fpl_starts.derived.rebuild.
    Evidence isn't in the base `predictions` SQL table's schema (that
    table stays minimal, per fpl_starts.derived's own scope), so this
    reads the snapshot files directly rather than querying derived.db,
    keeping only the latest snapshot per target_round the same way
    fpl_starts.derived._load_predictions does.

    One row per distinct evidence item (post-dedup, as predict.py already
    stored it) across every archived round for `model_version`. Returns
    the number of rows inserted.
    """
    conn.executescript(
        "DROP TABLE IF EXISTS agent_evidence;"
        "CREATE TABLE agent_evidence ("
        "  code INTEGER NOT NULL,"
        "  season TEXT NOT NULL,"
        "  target_round INTEGER NOT NULL,"
        "  category TEXT NOT NULL,"
        "  quote TEXT,"
        "  source_url TEXT,"
        "  domain TEXT"
        ");"
    )

    season_dir = os.path.join(predictions_dir, season)
    if not os.path.isdir(season_dir):
        return 0

    by_round = {}
    for name in os.listdir(season_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(season_dir, name)) as f:
            payload = json.load(f)
        if payload.get("model_version") != model_version:
            continue
        target_round = payload["target_round"]
        existing = by_round.get(target_round)
        if existing is None or payload["predicted_at"] > existing["predicted_at"]:
            by_round[target_round] = payload

    insert_sql = (
        "INSERT INTO agent_evidence "
        "(code, season, target_round, category, quote, source_url, domain) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    rows = []
    for target_round, payload in by_round.items():
        for row in payload["predictions"]:
            raw_evidence = row.get("evidence")
            if not raw_evidence:
                continue
            for item in json.loads(raw_evidence):
                rows.append((
                    row["code"], season, target_round, item["category"],
                    item.get("quote"), item.get("source_url"),
                    _domain_of(item.get("source_url")),
                ))
    conn.executemany(insert_sql, rows)
    conn.commit()
    return len(rows)


def fit_domain_rates(conn, season, target_round):
    """{domain: (n, n_correct)} of directional accuracy, from every round
    of `season` strictly before `target_round` -- walk-forward, same
    discipline as every other scoring function in this project. Empty on
    the first rounds this arm has ever run -- there's no history yet.
    """
    df = pd.read_sql(
        "SELECT ae.category AS category, ae.domain AS domain, pgs.starts AS y "
        "FROM agent_evidence ae "
        "JOIN player_gameweek_stats pgs "
        "  ON pgs.code = ae.code AND pgs.season = ae.season "
        "     AND pgs.round = ae.target_round "
        "WHERE ae.season = ? AND ae.target_round < ?",
        conn, params=(season, target_round),
    )
    df = df[df["category"].isin(DIRECTIONAL_CATEGORIES)]
    if len(df) == 0:
        return {}
    expected = df["category"].map(DIRECTIONAL_CATEGORIES)
    df = df.assign(correct=(df["y"].astype(int) == expected))
    grouped = df.groupby("domain")["correct"].agg(["size", "sum"])
    return dict(zip(grouped.index, zip(grouped["size"], grouped["sum"].astype(int))))


def _main():
    import argparse
    import sqlite3

    from fpl_starts import derived
    from fpl_starts import starts_model

    parser = argparse.ArgumentParser(
        description="Rebuild the agent's evidence table from archived "
                    "snapshots and report directional accuracy per source domain."
    )
    parser.add_argument("--db-path", default=derived.DERIVED_DB_PATH)
    parser.add_argument("--season", default=None,
                        help="Defaults to the only season in derived.db.")
    parser.add_argument("--predictions-dir", default=starts_model.PREDICTIONS_DIR)
    parser.add_argument("--target-round", type=int, default=None,
                        help="Score every round strictly before this one. "
                             "Defaults to (max archived round) + 1, i.e. "
                             "'everything archived so far counts as history'.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    season = args.season
    if season is None:
        seasons = pd.read_sql(
            "SELECT DISTINCT season FROM player_gameweek_stats", conn
        )["season"]
        if len(seasons) != 1:
            raise SystemExit(
                "derived.db has {0} seasons; pass --season explicitly".format(len(seasons))
            )
        season = seasons.iloc[0]

    target_round = args.target_round
    if target_round is None:
        max_round = pd.read_sql(
            "SELECT MAX(round) AS r FROM player_gameweek_stats WHERE season = ?",
            conn, params=(season,),
        )["r"].iloc[0]
        target_round = int(max_round) + 1 if max_round is not None else 1

    n_rows = rebuild_evidence_table(conn, args.predictions_dir, season)
    print("agent_evidence: {0} rows".format(n_rows))

    rates = fit_domain_rates(conn, season, target_round)
    conn.close()

    if not rates:
        print("no directional (confirmed_out/confirmed_starting) evidence "
              "scored against outcomes yet for {0} before round {1}".format(
                  season, target_round))
        return

    pd.set_option("display.width", 120)
    report = pd.DataFrame([
        {"domain": domain, "n": n, "n_correct": n_correct,
         "accuracy": n_correct / n}
        for domain, (n, n_correct) in rates.items()
    ]).sort_values("n", ascending=False).set_index("domain")
    print(report.round(3).to_string())


if __name__ == "__main__":
    _main()
