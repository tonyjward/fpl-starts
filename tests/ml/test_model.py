"""Preprocessing, fitting, persistence and explanation arithmetic."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fpl_starts.ml import logistic, spec
from fpl_starts.ml.preprocessing import Preprocessor, model_frame, sigmoid

from .conftest import training_frame


def test_fitting_refuses_prospective_season_rows():
    df = training_frame()
    df.loc[df.index[:3], "season"] = spec.PROSPECTIVE_SEASON
    with pytest.raises(logistic.TemporalBoundaryError):
        logistic.fit(df, 1.0)


def test_fitting_refuses_context_only_season_rows():
    df = training_frame()
    df.loc[df.index[:3], "season"] = spec.CONTEXT_ONLY_SEASONS[0]
    with pytest.raises(logistic.TemporalBoundaryError):
        logistic.fit(df, 1.0)


def test_tuning_refuses_prospective_season_validation_rows():
    train = training_frame(seasons=["2023-24"])
    validation = training_frame(seed=1, seasons=[spec.PROSPECTIVE_SEASON])
    with pytest.raises(logistic.TemporalBoundaryError):
        logistic.choose_C(train, validation)


def test_transformed_columns_follow_the_spec_and_drop_reference_levels():
    X = Preprocessor().fit(model_frame(training_frame())).transform(model_frame(training_frame(seed=3)))
    assert list(X.columns) == spec.transformed_feature_names()
    assert "availability_status__available" not in X.columns
    assert "last_gw_role__did_not_play" not in X.columns
    ref = training_frame(seed=3)
    is_ref = (ref["availability_status"] == "available").values
    avail_cols = [c for c in X.columns if c.startswith("availability_status__")]
    assert (X.loc[is_ref, avail_cols].values == 0).all()


def test_preprocessing_statistics_come_from_training_rows_only():
    train, other = training_frame(), training_frame(seed=9)
    other["minutes_prior_3_gws"] += 1000
    pre = Preprocessor().fit(model_frame(train))
    before = dict(pre.means), dict(pre.sds)
    pre.transform(model_frame(other))
    assert (pre.means, pre.sds) == before
    assert pre.means["minutes_prior_3_gws"] == pytest.approx(train["minutes_prior_3_gws"].mean())


def test_missing_continuous_value_transforms_to_zero():
    train = training_frame()
    pre = Preprocessor().fit(model_frame(train))
    row = model_frame(train.iloc[[0]]).copy()
    row["previous_season_start_rate"] = np.nan
    assert pre.transform(row)["previous_season_start_rate"].iloc[0] == 0.0


def test_unexpected_category_level_is_an_error():
    df = training_frame()
    pre = Preprocessor().fit(model_frame(df))
    bad = model_frame(df.iloc[[0]]).copy()
    bad["availability_status"] = "retired"
    with pytest.raises(ValueError, match="unexpected availability_status"):
        pre.transform(bad)


def test_fit_is_deterministic():
    a, b = logistic.fit(training_frame(), 1.0), logistic.fit(training_frame(), 1.0)
    assert a.intercept == b.intercept
    pd.testing.assert_series_equal(a.coefficients, b.coefficients)


def test_stored_coefficients_reproduce_the_fitted_estimator():
    df = training_frame()
    model = logistic.fit(df, 1.0)
    X = model.transform(df)
    est = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-8).fit(X.values, df["y"])
    np.testing.assert_allclose(model.predict_proba(df), est.predict_proba(X.values)[:, 1], atol=1e-10)
    assert list(model.coefficients.index) == list(X.columns) == spec.transformed_feature_names()


def test_contributions_sum_to_logit_and_sigmoid_gives_p_start():
    df = training_frame()
    model = logistic.fit(df, 1.0)
    explained = model.explain(df.head(25))
    np.testing.assert_allclose([e["p_start"] for e in explained], model.predict_proba(df.head(25)), atol=1e-12)
    for e in explained:
        assert [f["feature"] for f in e["features"]] == spec.transformed_feature_names()
        total = e["intercept"] + sum(f["contribution"] for f in e["features"])
        assert total == pytest.approx(e["logit"], abs=1e-9)
        assert sigmoid(e["logit"]) == pytest.approx(e["p_start"], abs=1e-12)
        for f in e["features"]:
            assert f["contribution"] == pytest.approx(f["coefficient"] * f["transformed_value"], abs=1e-12)


def test_explanation_keeps_raw_values_alongside_transformed_ones():
    df = training_frame()
    model = logistic.fit(df, 1.0)
    row = df.iloc[[0]]
    feats = {f["feature"]: f for f in model.explain(row)[0]["features"]}
    minutes = feats["minutes_prior_3_gws"]
    assert minutes["raw_value"] == row["minutes_prior_3_gws"].iloc[0]
    pre = model.preprocessor
    assert minutes["transformed_value"] == pytest.approx(
        (minutes["raw_value"] - pre.means["minutes_prior_3_gws"]) / pre.sds["minutes_prior_3_gws"])
    assert feats["last_gw_role__started_60_plus"]["raw_value"] == row["last_gw_role"].iloc[0]


def test_saved_model_round_trips_and_is_write_once(tmp_path):
    df = training_frame()
    model = logistic.fit(df, 0.3, metadata={"note": "test"})
    path = logistic.save(model, str(tmp_path))
    loaded = logistic.load(str(tmp_path))
    np.testing.assert_allclose(loaded.predict_proba(df), model.predict_proba(df), atol=1e-15)
    stored = json.load(open(path))
    assert set(stored["coefficients"]) == set(spec.transformed_feature_names())
    assert stored["transformed_features"] == spec.transformed_feature_names()
    assert stored["preprocessing"]["categorical"]["availability_status"]["reference"] == "available"
    with pytest.raises(FileExistsError):
        logistic.save(model, str(tmp_path))


def test_choose_c_picks_the_lowest_validation_brier():
    train, validation = training_frame(seasons=["2023-24"]), training_frame(seed=5, seasons=["2024-25"])
    best, scores = logistic.choose_C(train, validation, grid=[0.01, 1.0])
    assert scores[best] == min(scores.values())
