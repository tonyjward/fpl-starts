"""End-to-end training and historical evaluation on synthetic data: only
training-season rows are ever fitted or tuned on, and the model is frozen."""

import json
import os

import pandas as pd
import pytest

from fpl_starts.ml import evaluate, logistic, panel as mlpanel, spec, train


@pytest.fixture
def data_dir(world):
    world.fill_regular_season("2022-23", 12)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 12)
    world.write()
    return world.data_dir


def test_evaluation_ignores_prospective_and_context_rows(data_dir):
    panel = mlpanel.build_historical_panel(data_dir)
    result_a, _ = evaluate.run(panel)
    extra = panel[panel["season"] == "2025-26"].assign(season=spec.PROSPECTIVE_SEASON, y=1)
    flipped = panel.assign(y=lambda d: d["y"].where(d["season"] != "2022-23", 1 - d["y"]))
    result_b, _ = evaluate.run(pd.concat([flipped, extra], ignore_index=True))
    assert json.dumps(result_a, sort_keys=True) == json.dumps(result_b, sort_keys=True)


def test_train_fits_training_seasons_only_and_freezes(data_dir, tmp_path):
    models_dir = str(tmp_path / "models")
    model, path, _ = train.train(data_dir, models_dir)
    stored = json.load(open(path))
    md = stored["metadata"]
    assert md["training_seasons"] == list(spec.TRAINING_SEASONS)
    assert set(md["training_rows_by_season"]) == set(spec.TRAINING_SEASONS)
    assert md["C_selection"]["validation_season"] == "2025-26"
    assert md["C_selection"]["chosen_C"] in spec.C_GRID
    assert stored["transformed_features"] == spec.transformed_feature_names()
    assert os.path.isfile(os.path.join(os.path.dirname(path), "historical_evaluation.json"))
    with pytest.raises(FileExistsError):
        train.train(data_dir, models_dir)


def test_training_is_reproducible(data_dir, tmp_path):
    a, _, _ = train.train(data_dir, str(tmp_path / "a"))
    b, _, _ = train.train(data_dir, str(tmp_path / "b"))
    assert a.intercept == b.intercept
    pd.testing.assert_series_equal(a.coefficients, b.coefficients)
    assert a.metadata["training_matrix_sha256"] == b.metadata["training_matrix_sha256"]
