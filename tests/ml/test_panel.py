"""Historical panel construction: identity, exclusions, transfers, double
gameweeks and the no-look-ahead guarantee."""

import numpy as np
import pandas as pd
import pytest

from fpl_starts.ml import data as mldata
from fpl_starts.ml import panel as mlpanel
from fpl_starts.ml import spec


def _panel(world, seasons=("2022-23", "2023-24")):
    world.write()
    rows = mlpanel.load_historical_rows(world.data_dir, seasons)
    return mlpanel.with_features(rows, list(seasons))


def _row(panel, code, season, rnd):
    r = panel[(panel["code"] == code) & (panel["season"] == season) & (panel["round"] == rnd)]
    assert len(r) >= 1
    return r.iloc[0]


def test_player_identity_uses_code_not_season_local_id(world):
    # Same player (code 555) has element id 1 in 2022-23 and id 9 in 2023-24;
    # a different player reuses element id 1 in 2023-24.
    world.player("2022-23", 1, 555)
    world.player("2023-24", 9, 555)
    world.player("2023-24", 1, 777)
    for r in range(1, 5):
        world.appearance("2022-23", 1, r, "Alpha", 90, True)
        world.appearance("2023-24", 9, r, "Alpha", 90, True)
        world.appearance("2023-24", 1, r, "Alpha", 0, False)
    panel = _panel(world)
    assert _row(panel, 555, "2023-24", 1)["previous_season_start_rate"] == 1.0
    assert _row(panel, 555, "2023-24", 1)["no_previous_season"] == 0
    assert _row(panel, 777, "2023-24", 1)["no_previous_season"] == 1
    assert set(panel["code"]) == {555, 777}


def test_manager_rows_are_excluded(world):
    world.player("2022-23", 1, 555)
    world.player("2022-23", 2, 999, element_type=mldata.MANAGER_ELEMENT_TYPE)
    world.player("2023-24", 1, 555)
    for r in range(1, 3):
        world.appearance("2022-23", 1, r, "Alpha", 90, True)
        world.appearance("2022-23", 2, r, "Alpha", 0, False)
        world.appearance("2023-24", 1, r, "Alpha", 90, True)
    panel = _panel(world)
    assert 999 not in set(panel["code"])


def test_team_comes_from_the_fixture_row_and_a_transfer_resets_club_history(world):
    world.player("2022-23", 1, 555)
    world.player("2023-24", 1, 555)
    for r in range(1, 5):
        world.appearance("2022-23", 1, r, "Alpha", 90, True)
    for r in range(1, 7):
        team = "Alpha" if r <= 3 else "Bravo"  # transferred before GW4
        world.appearance("2023-24", 1, r, team, 90 if r <= 3 else 10, r <= 3)
    panel = _panel(world)
    assert [int(_row(panel, 555, "2023-24", r)["team_code"]) for r in range(1, 7)] == [101] * 3 + [102] * 3
    gw4 = _row(panel, 555, "2023-24", 4)
    assert gw4["first_game_at_club"] == 1
    assert pd.isna(gw4["current_season_start_rate"]) and pd.isna(gw4["minutes_prior_3_gws"])
    assert gw4["last_gw_role"] == "did_not_play"  # reference level; the flag carries the effect
    gw5 = _row(panel, 555, "2023-24", 5)
    assert gw5["first_game_at_club"] == 0
    assert gw5["current_season_start_rate"] == 0.0  # only Bravo gameweeks count
    assert gw5["last_gw_role"] == "sub_appearance"
    assert gw5["previous_season_start_rate"] == 1.0  # previous-season role carries across a transfer


def test_exact_duplicate_source_rows_are_dropped(world):
    world.player("2022-23", 1, 555)
    world.player("2023-24", 1, 555)
    for r in range(1, 4):
        world.appearance("2022-23", 1, r, "Alpha", 90, True)
        world.appearance("2023-24", 1, r, "Alpha", 45, True)
    world.appearance("2023-24", 1, 1, "Alpha", 45, True)  # byte-identical duplicate
    panel = _panel(world)
    assert len(panel[(panel["season"] == "2023-24") & (panel["round"] == 1)]) == 1
    assert _row(panel, 555, "2023-24", 2)["minutes_prev"] == 45


def test_double_gameweek_fixtures_share_one_feature_vector(world):
    world.player("2022-23", 1, 555)
    world.player("2023-24", 1, 555)
    for r in range(1, 4):
        world.appearance("2022-23", 1, r, "Alpha", 90, True)
    world.appearance("2023-24", 1, 1, "Alpha", 90, True)
    world.appearance("2023-24", 1, 2, "Alpha", 90, True, fixture=21)   # DGW: starts one ...
    world.appearance("2023-24", 1, 2, "Alpha", 20, False, fixture=22)  # ... comes off the bench in the other
    world.appearance("2023-24", 1, 3, "Alpha", 90, True)
    panel = _panel(world)
    dgw = panel[(panel["season"] == "2023-24") & (panel["round"] == 2)]
    assert len(dgw) == 2
    features = spec.RAW_FEATURES[1:] + ["prev_started", "minutes_prev", "stratum"]
    first, second = dgw.iloc[0], dgw.iloc[1]
    for f in features:
        assert (pd.isna(first[f]) and pd.isna(second[f])) or first[f] == second[f], f
    after = _row(panel, 555, "2023-24", 3)
    assert after["minutes_prev"] == 110  # both DGW fixtures' minutes
    assert after["last_gw_role"] == "started_60_plus"  # started at least one


def test_features_never_look_ahead(world):
    world.fill_regular_season("2022-23", 6)
    world.fill_regular_season("2023-24", 10)
    panel_a = _panel(world)
    # Rewrite every outcome from GW6 of 2023-24 onwards.
    world.gw_rows["2023-24"] = [dict(r, minutes=0, starts=0) if r["GW"] >= 6 else r
                                for r in world.gw_rows["2023-24"]]
    panel_b = _panel(world)
    cols = ["code", "season", "round", "fixture"] + spec.RAW_FEATURES[1:] + ["stratum"]
    early = lambda p: p[(p["season"] == "2022-23") | (p["round"] <= 6)][cols].sort_values(cols[:4]).reset_index(drop=True)
    pd.testing.assert_frame_equal(early(panel_a), early(panel_b))


def test_minutes_prior_3_gws_excludes_last_gameweek(world):
    world.player("2022-23", 1, 555)
    world.player("2023-24", 1, 555)
    world.appearance("2022-23", 1, 1, "Alpha", 90, True)
    for r, m in zip(range(1, 7), [10, 20, 30, 40, 50, 60]):
        world.appearance("2023-24", 1, r, "Alpha", m, m >= 30)
    panel = _panel(world)
    gw6 = _row(panel, 555, "2023-24", 6)
    assert gw6["minutes_prev"] == 50
    assert gw6["minutes_prior_3_gws"] == 20 + 30 + 40
    assert gw6["last_gw_role"] == "started_under_60"


@pytest.mark.parametrize("status,chance,expected", [
    ("a", None, "available"), ("a", 100, "available"), ("d", 75, "doubtful_75"),
    ("d", 50, "doubtful_50"), ("d", 25, "doubtful_25"), ("d", None, "doubtful_50"),
    ("i", 0, "injured"), ("s", 0, "suspended"), ("u", 0, "unavailable"), ("n", 0, "unavailable"),
    (None, None, "unknown"), (np.nan, np.nan, "unknown"),
])
def test_availability_status_levels(status, chance, expected):
    assert mlpanel.availability_status([status], [chance]) == [expected]


def test_unrecognised_status_is_an_error():
    with pytest.raises(ValueError):
        mlpanel.availability_status(["x"], [None])
