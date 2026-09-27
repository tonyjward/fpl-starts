"""The application-facing P(start) API (fpl_starts.pstart) and the dashboard's
data layer on top of it: frozen logistic model only, registered snapshots
read-only, live predictions without fitting or writing, clear failures and
explanations passed through intact. Synthetic inputs only."""

import ast
import importlib.util
import json
import os

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fpl_starts import derived, pstart
from fpl_starts.ml import logistic, predict, spec
from fpl_starts.ml.preprocessing import sigmoid

from ml.conftest import SyntheticWorld, add_current_gw, add_snapshot, deadline, training_frame

SEASON = spec.PROSPECTIVE_SEASON
CODES = list(range(1000, 1008))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")
RETIRED = ["raw_lookup", "refined_availability", "agent_news", "starts_model", "fpl_starts.agent"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    world = SyntheticWorld(str(tmp_path))
    world.fill_regular_season("2022-23", 6)
    for s in spec.TRAINING_SEASONS:
        world.fill_regular_season(s, 6)
    for r in (1, 2, 3, 4):
        for code in CODES:
            world.avail(SEASON, r, code)
    world.write()
    conn = world.db()
    for r in (1, 2, 3, 4):
        for i, code in enumerate(CODES):
            add_current_gw(conn, code, r, 101 + i % 2, 90 if i < 3 else 0, i < 3)
    cutoff5 = deadline(SEASON, 5) - pd.Timedelta(hours=spec.CUTOFF_HOURS_BEFORE_DEADLINE)
    for i, code in enumerate(CODES):
        add_snapshot(conn, code, 5, cutoff5 - pd.Timedelta(hours=6), team_code=101 + i % 2,
                     status="d" if code == 1001 else "a", chance=25 if code == 1001 else None)
    conn.executemany("INSERT INTO teams (code, name) VALUES (?, ?)", [(101, "Alpha"), (102, "Bravo")])
    conn.executemany("INSERT INTO players (code, player_id, web_name, team_code) VALUES (?, ?, ?, ?)",
                     [(code, i + 1, "Player{0}".format(i), 101 + i % 2) for i, code in enumerate(CODES)])
    conn.commit()
    conn.close()

    state = {"finished": 4}
    monkeypatch.setattr(derived, "latest_bootstrap_payload", lambda base, season: world.bootstrap(state["finished"]))
    models_dir = str(tmp_path / "models")
    logistic.save(logistic.fit(training_frame(), 1.0, metadata={"created_at": "2026-08-01T00:00:00Z"}),
                  logistic.model_dir(models_dir))
    return {"world": world, "db": world.db_path, "data_dir": world.data_dir, "models_dir": models_dir,
            "predictions_dir": str(tmp_path / "predictions"), "state": state}


def _live(setup, target_round=5, model=None):
    model = model or pstart.load_frozen_model(setup["models_dir"])
    return pstart.predict_logistic_p_start(model, SEASON, target_round, setup["db"], setup["data_dir"], "raw")


def _register(setup, target_round=5, when="2026-09-18 10:00"):
    """A registered snapshot, written the way fpl-starts-logistic-predict writes one."""
    import sqlite3
    model = pstart.load_frozen_model(setup["models_dir"])
    conn = sqlite3.connect(setup["db"])
    records, detail, cutoff, dl = predict.predict_round(conn, setup["data_dir"], "raw", model, SEASON, target_round)
    conn.close()
    clock = lambda: pd.Timestamp(when, tz="UTC").to_pydatetime()
    path = predict.write_snapshot(records, detail, model, pstart.model_path(setup["models_dir"]), SEASON,
                                  target_round, cutoff, dl, base_dir=setup["predictions_dir"], clock=clock)
    return path, records, detail


def _registered(setup, target_round=5):
    return pstart.load_registered_predictions(SEASON, target_round, setup["predictions_dir"], setup["db"])


# --- 1, 2: logistic probabilities from logistic_availability_v1 -----------------

def test_live_predictions_are_the_frozen_logistic_models(setup):
    result = _live(setup)
    players = result.players.set_index("code")
    assert sorted(players.index) == CODES
    assert players["p_start"].between(0, 1).all()
    assert players["p_start"].tolist() == pytest.approx(sigmoid(players["logit"]).tolist())
    assert players.loc[1001, "availability_status"] == "doubtful_25"
    assert players.loc[1000, "last_gw_role"] == "started_60_plus"
    assert players.loc[1000, "current_season_start_rate"] == 1.0
    assert players.loc[1000, "web_name"] == "Player0" and players.loc[1000, "team"] == "Alpha"
    assert list(result.players.columns) == pstart.PLAYER_COLUMNS
    assert result.metadata["model_id"] == "logistic_availability_v1"
    assert result.metadata["source"] == pstart.SOURCE_LIVE


def test_registered_and_live_agree_for_the_same_inputs(setup):
    _, records, _ = _register(setup)
    registered = _registered(setup)
    live = _live(setup)
    assert registered.metadata["model_id"] == spec.MODEL_ID == "logistic_availability_v1"
    assert registered.metadata["source"] == pstart.SOURCE_REGISTERED
    pd.testing.assert_frame_equal(registered.players, live.players)
    assert registered.players.set_index("code")["p_start"].to_dict() == {r["code"]: r["p_start"] for r in records}


def test_latest_registered_snapshot_is_used(setup):
    _register(setup, when="2026-09-18 09:00")
    _register(setup, when="2026-09-18 10:00")
    assert _registered(setup).metadata["predicted_at"] == "20260918T100000Z"


def test_load_frozen_model_rejects_a_different_model_id(setup, tmp_path):
    stored = json.load(open(pstart.model_path(setup["models_dir"])))
    stored["model_id"] = "logistic_availability_v2"
    other = str(tmp_path / "other")
    os.makedirs(logistic.model_dir(other))
    json.dump(stored, open(pstart.model_path(other), "w"))
    with pytest.raises(pstart.ModelArtefactNotFound, match="not logistic_availability_v1"):
        pstart.load_frozen_model(other)


# --- 3: no retired-model fallback -------------------------------------------------

def _write_payload(setup, name, payload):
    d = os.path.join(setup["predictions_dir"], SEASON)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        json.dump(payload, f)


def test_retired_model_snapshots_are_never_used(setup):
    for version in ("raw_lookup", "refined_availability", "refined_availability_agent_news"):
        _write_payload(setup, "gw05_{0}_20260918T100000Z.json".format(version), {
            "season": SEASON, "target_round": 5, "model_version": version, "predicted_at": "20260918T100000Z",
            "predictions": [{"code": c, "p_start": 0.5, "cold_start": False, "n_observed": 1, "method": version}
                            for c in CODES]})
    with pytest.raises(pstart.NoPredictionAvailable, match="No prediction available for this gameweek"):
        _registered(setup)


def test_snapshot_from_another_logistic_model_id_is_refused(setup):
    path, _, _ = _register(setup)
    payload = json.load(open(path))
    payload["model"]["model_id"] = "logistic_availability_v2"
    payload["predicted_at"] = "20260919T000000Z"
    _write_payload(setup, "gw05_logistic_availability_20260919T000000Z.json", payload)
    with pytest.raises(pstart.NoPredictionAvailable, match="not logistic_availability_v1"):
        _registered(setup)


def test_no_retired_model_is_referenced_by_the_app_or_service():
    sources = [os.path.join(REPO_ROOT, "src", "fpl_starts", "pstart.py")] + [
        os.path.join(DASHBOARD_DIR, name) for name in ("app.py", "data.py", "agent.py")]
    for path in sources:
        text = open(path).read()
        for name in RETIRED:
            assert name not in text, "{0} references retired {1}".format(path, name)


# --- 4: loaded, never retrained or written ------------------------------------------

def test_predictions_never_fit_or_write(setup, monkeypatch):
    _register(setup)

    def boom(*a, **k):
        raise AssertionError("the dashboard path must not fit or write anything")
    for target, attr in [(LogisticRegression, "fit"), (logistic, "fit"), (logistic, "save"),
                         (predict, "write_snapshot")]:
        monkeypatch.setattr(target, attr, boom)
    model_sha = logistic.file_sha256(pstart.model_path(setup["models_dir"]))
    before = sorted(os.listdir(os.path.join(setup["predictions_dir"], SEASON)))

    _live(setup)
    _registered(setup)

    assert logistic.file_sha256(pstart.model_path(setup["models_dir"])) == model_sha
    assert sorted(os.listdir(os.path.join(setup["predictions_dir"], SEASON))) == before


# --- 5: clear failures ------------------------------------------------------------------

def test_missing_model_artefact(tmp_path):
    with pytest.raises(pstart.ModelArtefactNotFound, match="Logistic model artefact not found"):
        pstart.load_frozen_model(str(tmp_path / "nowhere"))


def test_missing_database(setup, tmp_path):
    model = pstart.load_frozen_model(setup["models_dir"])
    with pytest.raises(pstart.InputDataUnavailable, match="Current availability data unavailable"):
        pstart.predict_logistic_p_start(model, SEASON, 5, str(tmp_path / "none.db"), setup["data_dir"], "raw")


def test_missing_local_availability_data(setup):
    os.remove(os.path.join(setup["data_dir"], "availability", "gw3_gw4_availability.csv"))
    with pytest.raises(pstart.InputDataUnavailable, match="Current availability data unavailable"):
        _live(setup)


def test_no_registered_prediction_for_the_gameweek(setup):
    _register(setup)
    with pytest.raises(pstart.NoPredictionAvailable, match="No prediction available for this gameweek"):
        _registered(setup, target_round=6)


def test_live_prediction_refused_before_earlier_gameweeks_finish(setup):
    setup["state"]["finished"] = 3
    with pytest.raises(pstart.NoPredictionAvailable, match="not finished"):
        _live(setup)


# --- 6: explanations passed through -----------------------------------------------------

def test_contributions_are_passed_through_exactly(setup):
    _, records, detail = _register(setup)
    result = _registered(setup)
    c = result.contributions
    assert len(c) == len(CODES) * len(spec.transformed_feature_names())
    by_code = {e["code"]: e for e in detail}
    for code, group in c.groupby("code"):
        exp = by_code[code]
        assert group["feature"].tolist() == [f["feature"] for f in exp["features"]]
        assert group["contribution"].tolist() == [f["contribution"] for f in exp["features"]]
        assert group["coefficient"].tolist() == [f["coefficient"] for f in exp["features"]]
        assert result.metadata["intercept"] + group["contribution"].sum() == pytest.approx(exp["logit"])
    assert c["description"].notna().all()


def test_top_factors_split_by_sign_and_rank_by_size(setup):
    result = _live(setup)
    positive, negative = result.top_factors(1001, n=3)
    assert (positive["contribution"] > 0).all() and (negative["contribution"] < 0).all()
    assert positive["contribution"].is_monotonic_decreasing and negative["contribution"].is_monotonic_increasing
    assert "availability_status__doubtful_25" in negative["feature"].tolist()


# --- 7: public-only imports ----------------------------------------------------------------

ALLOWED_IMPORTS = {"__future__", "dataclasses", "json", "os", "sqlite3", "pandas", "requests", "fpl_starts"}


def _imported_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_service_and_dashboard_data_import_only_public_packages():
    for path in (os.path.join(REPO_ROOT, "src", "fpl_starts", "pstart.py"), os.path.join(DASHBOARD_DIR, "data.py")):
        assert _imported_roots(path) <= ALLOWED_IMPORTS, path
    for name in ("app.py", "data.py", "agent.py", "README.md"):
        text = open(os.path.join(DASHBOARD_DIR, name)).read()
        assert os.path.join("..", "fpl", "") not in text and "FPL_DASHBOARD_NEWS_DB" not in text, name


# --- dashboard data layer ------------------------------------------------------------------

@pytest.fixture
def dashboard_data(setup, monkeypatch):
    spec_ = importlib.util.spec_from_file_location("dashboard_data", os.path.join(DASHBOARD_DIR, "data.py"))
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    monkeypatch.setattr(module, "FPL_STARTS_DB_PATH", setup["db"])
    monkeypatch.setattr(module, "MODELS_DIR", setup["models_dir"])
    monkeypatch.setattr(module, "PREDICTIONS_DIR", setup["predictions_dir"])
    monkeypatch.setattr(module, "DATA_DIR", setup["data_dir"])
    return module


def test_dashboard_reads_registered_and_live_logistic_predictions(setup, dashboard_data):
    _register(setup)
    registered = dashboard_data.load_gameweek_predictions(SEASON, 5)
    assert registered.metadata["source"] == pstart.SOURCE_REGISTERED
    assert registered.metadata["model_id"] == spec.MODEL_ID
    live = dashboard_data.load_gameweek_predictions(SEASON, 5, dashboard_data.SOURCE_LIVE)
    assert live.metadata["source"] == pstart.SOURCE_LIVE
    with pytest.raises(ValueError, match="unknown P\\(start\\) source"):
        dashboard_data.load_gameweek_predictions(SEASON, 5, "raw_lookup")


def test_dashboard_top_factor_columns(setup, dashboard_data):
    players = dashboard_data.with_top_factors(_live(setup))
    row = players.set_index("code").loc[1001]
    assert "availability_status__doubtful_25" in row["top_negative"]
    assert row["top_positive"] == "" or row["top_positive"].startswith("+")


def test_dashboard_squad_uses_logistic_p_start(setup, dashboard_data, monkeypatch):
    _register(setup)
    monkeypatch.setattr(dashboard_data, "fetch_bootstrap",
                        lambda: {"elements": [{"id": i + 1, "code": c} for i, c in enumerate(CODES)]})
    monkeypatch.setattr(dashboard_data, "fetch_team_picks", lambda team_id, event: {"picks": [
        {"element": 1, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
        {"element": 2, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": True}]})
    squad = dashboard_data.load_squad_predictions(1, 5, SEASON, 5)
    expected = _registered(setup).players.set_index("code")["p_start"]
    assert squad["p_start"].tolist() == [expected[1000], expected[1001]]
    assert squad["availability_status"].tolist() == ["available", "doubtful_25"]
