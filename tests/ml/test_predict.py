"""Prospective scoring with the frozen model: no refitting, pre-cutoff inputs
only, explanation-ready and write-once output."""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fpl_starts import derived
from fpl_starts.ml import logistic, predict, spec
from fpl_starts.ml.preprocessing import sigmoid

from .conftest import add_current_gw, add_snapshot, deadline, training_frame

SEASON = spec.PROSPECTIVE_SEASON
CODES = list(range(1000, 1008))


@pytest.fixture
def setup(world, tmp_path, monkeypatch):
    world.fill_regular_season("2022-23", 6)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 6)
    for r in (1, 2, 3, 4):
        for i, code in enumerate(CODES):
            world.avail(SEASON, r, code, status="d" if (code == 1001 and r == 3) else "a",
                        chance=50 if (code == 1001 and r == 3) else None)
    world.write()
    conn = world.db()
    for r in (1, 2, 3, 4):
        for i, code in enumerate(CODES):
            add_current_gw(conn, code, r, 101 + i % 2, 90 if i < 3 else 0, i < 3)
    cutoff5 = deadline(SEASON, 5) - pd.Timedelta(hours=spec.CUTOFF_HOURS_BEFORE_DEADLINE)
    for i, code in enumerate(CODES):
        add_snapshot(conn, code, 5, cutoff5 - pd.Timedelta(hours=6), team_code=101 + i % 2)
    add_snapshot(conn, 1000, 5, cutoff5 + pd.Timedelta(hours=1), status="i", chance=0)  # after cutoff
    conn.commit()

    state = {"finished": 4}
    monkeypatch.setattr(derived, "latest_bootstrap_payload", lambda base, season: world.bootstrap(state["finished"]))
    model_dir = str(tmp_path / "models")
    model = logistic.fit(training_frame(), 1.0, metadata={"created_at": "2026-08-01T00:00:00Z",
                                                          "training_seasons": list(spec.TRAINING_SEASONS)})
    logistic.save(model, model_dir)
    return {"world": world, "conn": conn, "model": logistic.load(model_dir),
            "model_path": os.path.join(model_dir, "model.json"), "state": state}


def _predict(setup, target_round):
    return predict.predict_round(setup["conn"], setup["world"].data_dir, "raw", setup["model"], SEASON, target_round)


def test_scoring_never_refits(setup, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("prediction must not fit anything")
    monkeypatch.setattr(LogisticRegression, "fit", boom)
    monkeypatch.setattr(logistic, "fit", boom)
    records, detail, _, _ = _predict(setup, 5)
    assert sorted(r["code"] for r in records) == CODES


def test_explanations_are_exact(setup):
    records, detail, _, _ = _predict(setup, 5)
    for rec, exp in zip(records, detail):
        assert rec["code"] == exp["code"]
        assert [f["feature"] for f in exp["features"]] == spec.transformed_feature_names()
        assert exp["intercept"] + sum(f["contribution"] for f in exp["features"]) == pytest.approx(exp["logit"])
        assert sigmoid(exp["logit"]) == pytest.approx(rec["p_start"])
        assert rec["logit"] == exp["logit"]


def test_post_cutoff_snapshot_is_ignored(setup):
    records, _, _, _ = _predict(setup, 5)
    assert {r["code"]: r["availability_status"] for r in records}[1000] == "available"


def test_local_file_rounds_are_scored_from_pre_cutoff_files(setup):
    records, _, cutoff, _ = _predict(setup, 3)
    status = {r["code"]: r["availability_status"] for r in records}
    assert status[1001] == "doubtful_50" and status[1000] == "available"


def test_features_use_only_earlier_gameweeks(setup):
    before, _, _, _ = _predict(setup, 5)
    # Outcomes from GW5 onwards appearing in the archive must not change GW5's predictions.
    for code in CODES:
        add_current_gw(setup["conn"], code, 5, 101, 0, False)
        add_current_gw(setup["conn"], code, 6, 101, 0, False)
    after, _, _, _ = _predict(setup, 5)
    assert [r["p_start"] for r in before] == [r["p_start"] for r in after]


def test_last_gameweek_role_reflects_the_previous_gameweek(setup):
    records, detail, _, _ = _predict(setup, 5)
    roles = {r["code"]: r["last_gw_role"] for r in records}
    assert roles[1000] == "started_60_plus" and roles[1007] == "did_not_play"


def test_refuses_when_an_earlier_gameweek_is_unfinished(setup):
    setup["state"]["finished"] = 3
    with pytest.raises(ValueError, match="not finished"):
        _predict(setup, 5)


def test_only_the_prospective_season_is_scored(setup):
    with pytest.raises(ValueError, match="scores"):
        predict.predict_round(setup["conn"], setup["world"].data_dir, "raw", setup["model"], "2025-26", 5)


def test_snapshot_is_write_once_and_loads_into_the_predictions_table(setup, tmp_path):
    records, detail, cutoff, dl = _predict(setup, 5)
    base = str(tmp_path / "predictions")
    clock = lambda: pd.Timestamp("2026-09-18 10:00", tz="UTC").to_pydatetime()
    path = predict.write_snapshot(records, detail, setup["model"], setup["model_path"], SEASON, 5, cutoff, dl,
                                  base_dir=base, clock=clock)
    with pytest.raises(FileExistsError):
        predict.write_snapshot(records, detail, setup["model"], setup["model_path"], SEASON, 5, cutoff, dl,
                               base_dir=base, clock=clock)
    payload = json.load(open(path))
    assert payload["model_version"] == spec.MODEL_VERSION
    assert payload["model"]["model_id"] == spec.MODEL_ID
    assert payload["model"]["model_sha256"] == logistic.file_sha256(setup["model_path"])
    assert payload["generated_after_deadline"] is False
    assert len(payload["explanations"]) == len(payload["predictions"])

    conn = sqlite3.connect(":memory:")
    conn.executescript(derived.BASE_SCHEMA)
    derived._load_predictions(conn, base, SEASON)
    got = pd.read_sql("SELECT code, p_start, model_version FROM predictions", conn)
    assert set(got["model_version"]) == {spec.MODEL_VERSION}
    expected = [r["p_start"] for r in sorted(records, key=lambda r: r["code"])]
    np.testing.assert_allclose(got.sort_values("code")["p_start"], expected)


def test_snapshot_generated_after_the_deadline_is_marked(setup, tmp_path):
    records, detail, cutoff, dl = _predict(setup, 5)
    clock = lambda: pd.Timestamp("2026-12-01", tz="UTC").to_pydatetime()
    path = predict.write_snapshot(records, detail, setup["model"], setup["model_path"], SEASON, 5, cutoff, dl,
                                  base_dir=str(tmp_path / "p"), clock=clock)
    assert json.load(open(path))["generated_after_deadline"] is True
