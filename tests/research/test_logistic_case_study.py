"""The case-study helpers: leakage-safe candidate features, an in-memory
specification swap that always restores `spec`, and the deterministic
worked-example rules. Synthetic data only."""

import math

import numpy as np
import pandas as pd
import pytest

from fpl_starts.ml import spec
from fpl_starts.research import logistic_case_study as cs


def _panel(started, minutes, team_stint=None, season="2023-24"):
    n = len(started)
    df = pd.DataFrame({
        "code": 1, "season": season, "round": range(1, n + 1), "period": range(1, n + 1),
        "team_stint": team_stint or [0] * n, "y": started, "minutes": minutes,
        "availability_status": "available", "chance_of_playing_next_round": np.nan,
    })
    df["prev_started"] = df.groupby("team_stint")["y"].shift(1)
    return df


def test_candidates_use_only_earlier_gameweeks():
    out = cs.add_candidate_features(_panel([1, 1, 0, 1, 1], [90, 80, 0, 90, 30]))
    assert out["starts_last_4"].tolist()[1:] == [1, 2, 2, 3]
    assert np.isnan(out["starts_last_4"].iloc[0])
    assert out["minutes_last_1"].tolist()[1:] == [90, 80, 0, 90]
    assert out["consecutive_starts"].tolist()[1:] == [1, 2, 0, 1]
    assert out["minutes_last_4"].iloc[4] == 90 + 80 + 0 + 90


def test_candidates_reset_on_a_change_of_club():
    out = cs.add_candidate_features(_panel([1, 1, 1, 1], [90, 90, 90, 90], team_stint=[0, 0, 1, 1]))
    assert np.isnan(out["starts_last_4"].iloc[2])
    assert out["starts_last_4"].iloc[3] == 1


def test_chance_of_playing_is_100_when_available_without_a_chance():
    out = cs.add_candidate_features(_panel([1, 1], [90, 90]))
    assert out["chance_of_playing"].tolist() == [100.0, 100.0]
    assert out["is_available"].tolist() == [1, 1]


def test_specification_swap_is_restored_even_on_error():
    before = (list(spec.CONTINUOUS), list(spec.BINARY), dict(spec.CATEGORICAL), list(spec.RAW_FEATURES))
    with pytest.raises(RuntimeError):
        with cs.specification(cs.MINUTES_LAST_4):
            assert spec.CONTINUOUS[0] == "minutes_last_4"
            raise RuntimeError
    assert (spec.CONTINUOUS, spec.BINARY, spec.CATEGORICAL, spec.RAW_FEATURES) == before


# --- Worked-example selection ----------------------------------------------------

def _explanation(code, status="available", role="started_60_plus", others=1.0, rate=1.0):
    """A self-consistent stored explanation: `others` is the combined
    contribution of the non-availability columns."""
    coef = {"availability_status__" + s: -1.0 * (i + 1) for i, s in enumerate(cs.FLAG_PREFERENCE)}
    features, intercept = [], -2.0
    for name in spec.transformed_feature_names():
        raw = name.split("__")[0]
        if raw == "availability_status":
            on = name == "availability_status__" + status
            c = coef.get(name, -3.0)
            features.append({"feature": name, "raw_feature": raw, "raw_value": status,
                             "coefficient": c, "transformed_value": float(on), "contribution": c * on})
        else:
            value = {"current_season_start_rate": rate, "last_gw_role": role}.get(raw, 0.0)
            contribution = others if name == "minutes_prior_3_gws" else 0.0
            features.append({"feature": name, "raw_feature": raw, "raw_value": value,
                             "coefficient": 1.0, "transformed_value": contribution, "contribution": contribution})
    logit = intercept + sum(f["contribution"] for f in features)
    return {"code": code, "features": features, "intercept": intercept, "logit": logit,
            "p_start": 1 / (1 + math.exp(-logit))}


def _snapshot(gw, rows):
    explanations = [_explanation(**r) for r in rows]
    predictions = [{"code": e["code"], "availability_status": r.get("status", "available"),
                    "last_gw_role": r.get("role", "started_60_plus"), "p_start": e["p_start"]}
                   for e, r in zip(explanations, rows)]
    return {"target_round": gw, "explanations": explanations, "predictions": predictions}


def test_flagged_example_prefers_the_most_graded_doubtful_then_the_strongest_history():
    snapshots = {
        1: _snapshot(1, [{"code": 1, "status": "injured", "others": 9.0},
                         {"code": 2, "status": "doubtful_50", "others": 3.0},
                         {"code": 3, "status": "doubtful_50", "others": 5.0}]),
        2: _snapshot(2, [{"code": 4, "status": "doubtful_75", "others": 8.0},
                         {"code": 5, "status": "unknown", "others": 9.0}]),
    }
    pick = cs.select_flagged_example(snapshots)
    assert (pick["gameweek"], pick["code"], pick["status"]) == (1, 3, "doubtful_50")
    assert pick["p_without_availability"] == pytest.approx(1 / (1 + math.exp(-(-2.0 + 5.0))))


def test_flagged_example_skips_incomplete_explanations():
    snap = _snapshot(1, [{"code": 1, "status": "doubtful_25", "others": 5.0},
                         {"code": 2, "status": "doubtful_25", "others": 1.0}])
    snap["explanations"][0]["logit"] += 1.0  # stored arithmetic no longer adds up
    assert cs.select_flagged_example({1: snap})["code"] == 2


def test_regular_starter_prefers_the_named_player_in_the_latest_eligible_gameweek():
    snapshots = {5: _snapshot(5, [{"code": 7, "others": 1.0}, {"code": 8, "others": 3.0}]),
                 6: _snapshot(6, [{"code": 7, "role": "did_not_play"}, {"code": 8, "others": 3.0}])}
    names = {7: ("Haaland", "MCI"), 8: ("Other", "XXX")}
    assert cs.select_regular_starter(snapshots, names) == (5, 7)


def test_regular_starter_falls_back_to_the_highest_probability_eligible_player():
    snapshots = {6: _snapshot(6, [{"code": 7, "others": 1.0}, {"code": 8, "others": 3.0},
                                  {"code": 9, "others": 5.0, "rate": None}])}
    assert cs.select_regular_starter(snapshots, {}) == (6, 8)


def test_start_rate_bands():
    assert cs.band(0, 0) == "No gameweek yet this season"
    assert cs.band(0, 5) == "Never started this season"
    assert cs.band(1, 10) == "Started under 15%"
    assert cs.band(2, 10) == "Started 15-50%"
    assert cs.band(5, 10) == "Started 50-90%"
    assert cs.band(9, 10) == "Started 90%+"


def test_season_so_far_uses_only_earlier_gameweeks():
    panel = pd.DataFrame({"code": 1, "season": "2023-24", "round": [1, 2, 3, 3], "y": [1, 0, 1, 1]})
    out = cs._season_so_far(panel).set_index("round")
    assert out.loc[1, "games_before"] == 0 and out.loc[3, "games_before"] == 2
    assert out.loc[3, "starts_before"] == 1


def test_naive_lookup_matches_the_benchmark_fit():
    rows = []
    for season in ("2023-24", "2024-25"):
        for code, prev, y in [(1, 1.0, 1), (2, 1.0, 0), (3, 0.0, 0), (4, np.nan, 1)]:
            rows.append({"season": season, "code": code, "prev_started": prev, "y": y})
    table = cs.naive_lookup(pd.DataFrame(rows))
    cells = {c["cell"]: c["p_start"] for c in table["2024-25"]["cells"]}
    assert cells == {"started last gameweek": 0.5, "did not start last gameweek": 0.0,
                     "first gameweek at club": 1.0}
    assert table["2025-26"]["train_seasons"] == ["2023-24", "2024-25"]
