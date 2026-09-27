"""The application-facing P(start) API (fpl_starts.pstart) and the dashboard's
data layer on top of it: frozen logistic model only, registered snapshots
read-only, live predictions without fitting or writing, clear failures and
explanations passed through intact. Synthetic inputs only."""

import ast
import importlib.util
import json
import math
import os
import sqlite3

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fpl_starts import derived, explanation, pstart
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
    cutoff5 = deadline(SEASON, 5) - pd.Timedelta(hours=predict.FORECAST_CUTOFF_HOURS_BEFORE_DEADLINE)
    for i, code in enumerate(CODES):
        add_snapshot(conn, code, 5, cutoff5 - pd.Timedelta(hours=6), team_code=101 + i % 2,
                     status="d" if code == 1001 else "a", chance=25 if code == 1001 else None)
    conn.executemany("INSERT INTO teams (code, name) VALUES (?, ?)", [(101, "Alpha"), (102, "Bravo")])
    conn.executemany("INSERT INTO players (code, player_id, web_name, team_code, element_type, first_name, "
                     "second_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [(code, i + 1, "Player{0}".format(i), 101 + i % 2, 2 if i < 4 else 3, "First{0}".format(i),
                       "Player{0}".format(i)) for i, code in enumerate(CODES)])
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
    return pstart.load_registered_predictions(SEASON, target_round, setup["predictions_dir"], setup["db"],
                                              setup["models_dir"])


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


# --- grouped explanation, compared with a nailed-on starter ---------------------------------

def _reference_logit(result):
    p = result.metadata["reference_p_start"]
    return math.log(p / (1 - p))


def test_groups_cover_every_model_input_exactly_once():
    features = [f for group in explanation.GROUPS.values() for f in group]
    assert sorted(features) == sorted(spec.RAW_FEATURES) and len(features) == len(set(features))
    assert set(explanation.NAILED_ON) == set(spec.RAW_FEATURES)


@pytest.mark.parametrize("source", ["live", "registered"])
def test_group_effects_reproduce_every_prediction_exactly(setup, source):
    if source == "registered":
        _register(setup)
    result = _live(setup) if source == "live" else _registered(setup)
    totals = result.explanation.groupby("code")["effect"].sum() + _reference_logit(result)
    logits = result.players.set_index("code")["logit"]
    assert totals.loc[logits.index].tolist() == pytest.approx(logits.tolist(), abs=1e-9)
    assert result.metadata["reference_p_start"] == pytest.approx(
        sigmoid(explanation.reference_values(pstart.load_frozen_model(setup["models_dir"]))[1]))


def test_what_if_figures_are_the_models_prediction_for_a_real_row(setup):
    """Each group's "if like a nailed-on starter" chance is exactly what the
    frozen model predicts for the player with that whole group replaced --
    a full, consistent set of inputs, not one input changed on its own."""
    model = pstart.load_frozen_model(setup["models_dir"])
    result = _live(setup)
    raw = {code: dict(zip(g["raw_feature"], g["raw_value"])) for code, g in result.contributions.groupby("code")}
    for row in result.explanation.itertuples():
        inputs = dict(raw[row.code])
        for feature in explanation.GROUPS[row.group] if row.group in explanation.GROUPS else (
                explanation.GROUPS[explanation.CLUB_PLAYING_TIME] + explanation.GROUPS[explanation.LAST_SEASON]):
            inputs[feature] = explanation.NAILED_ON[feature]
        frame = pd.DataFrame([inputs]).astype({"minutes_prior_3_gws": float, "current_season_start_rate": float,
                                              "previous_season_start_rate": float})
        assert model.predict_proba(frame)[0] == pytest.approx(row.p_start_if_nailed_on, abs=1e-12)


def test_a_nailed_on_player_has_nothing_holding_him_back(setup):
    result = _live(setup)
    regular = result.explain(1000)  # started every game, 90 minutes, available, 100% last season
    assert regular["effect"].abs().max() < 1e-9
    assert explanation.summary(regular) == "nothing is holding him back compared with a nailed-on starter"
    assert result.players.set_index("code").loc[1000, "p_start"] == pytest.approx(result.metadata["reference_p_start"])


def test_explanation_facts_and_ordering(setup):
    result = _live(setup)
    doubtful = result.explain(1001)
    assert doubtful["effect"].is_monotonic_increasing  # most-limiting group first
    facts = dict(zip(doubtful["label"], doubtful["facts"]))
    assert facts["Availability"] == "flagged doubtful (25% chance of playing)"
    assert facts["Playing time at his club"].startswith("played 60+ minutes last gameweek")
    squad_player = result.explain(1007)  # never started this season
    club = squad_player.set_index("label").loc["Playing time at his club"]
    assert "didn't play last gameweek" in club["facts"] and "started 0% of his games this season" in club["facts"]
    assert club["effect"] < 0 and club["p_start_if_nailed_on"] > result.players.set_index("code").loc[1007, "p_start"]
    assert "held back by playing time at his club" in explanation.summary(squad_player)


def test_early_gameweeks_merge_the_playing_time_groups(setup):
    early = _live(setup, target_round=3)
    assert set(early.explanation["label"]) == {"Availability", "Playing time"}
    later = _live(setup, target_round=5)
    assert set(later.explanation["label"]) == {"Availability", "Playing time at his club", "Last season"}


def test_snapshot_from_a_different_model_file_is_refused(setup):
    path, _, _ = _register(setup)
    payload = json.load(open(path))
    payload["model"]["model_sha256"] = "0" * 64
    json.dump(payload, open(path, "w"))
    with pytest.raises(pstart.NoPredictionAvailable, match="different logistic_availability_v1 model file"):
        _registered(setup)


def test_registered_predictions_need_the_frozen_model(setup, tmp_path):
    _register(setup)
    with pytest.raises(pstart.ModelArtefactNotFound, match="Logistic model artefact not found"):
        pstart.load_registered_predictions(SEASON, 5, setup["predictions_dir"], setup["db"], str(tmp_path / "none"))


# --- 7: public-only imports ----------------------------------------------------------------

ALLOWED_IMPORTS = {"__future__", "dataclasses", "json", "math", "os", "sqlite3", "pandas", "requests", "fpl_starts"}


def _imported_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_service_and_dashboard_data_import_only_public_packages():
    for path in (os.path.join(REPO_ROOT, "src", "fpl_starts", "pstart.py"),
                 os.path.join(REPO_ROOT, "src", "fpl_starts", "explanation.py"), os.path.join(DASHBOARD_DIR, "data.py")):
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


def test_dashboard_reads_players_and_last_gameweek_from_derived_db(setup, dashboard_data):
    players = dashboard_data.load_player_universe()
    assert sorted(players) == CODES
    assert players[1000] == {"code": 1000, "element": 1, "web_name": "Player0", "full_name": "First0 Player0",
                             "display_name": "First0 Player0", "known_name": "", "second_name": "Player0",
                             "team": "Alpha", "position": "DEF"}
    conn = sqlite3.connect(setup["db"])
    conn.execute("UPDATE players SET known_name = 'Known Nought' WHERE code = 1000")
    conn.commit()
    conn.close()
    renamed = dashboard_data.load_player_universe()[1000]
    assert renamed["display_name"] == "Known Nought" and renamed["full_name"] == "First0 Player0"
    assert players[1005]["position"] == "MID" and players[1005]["team"] == "Bravo"
    assert dashboard_data.last_completed_gameweek(SEASON) == 4
    assert dashboard_data.last_completed_gameweek("2030-31") is None
    assert dashboard_data.db_version() == os.path.getmtime(setup["db"])


def test_dashboard_player_list_needs_a_database_with_names(setup, dashboard_data, tmp_path, monkeypatch):
    old = str(tmp_path / "old.db")
    sqlite3.connect(old).executescript("CREATE TABLE teams (code INTEGER, name TEXT);"
                                       "CREATE TABLE players (code INTEGER, player_id INTEGER, web_name TEXT, "
                                       "team_code INTEGER, element_type INTEGER);")
    monkeypatch.setattr(dashboard_data, "FPL_STARTS_DB_PATH", old)
    with pytest.raises(RuntimeError, match="rebuild it with fpl-starts-derive"):
        dashboard_data.load_player_universe()


def test_dashboard_squad_uses_logistic_p_start(setup, dashboard_data, monkeypatch):
    _register(setup)
    monkeypatch.setattr(dashboard_data, "fetch_team_picks", lambda team_id, event: {"picks": [
        {"element": 1, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
        {"element": 2, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": True}]})
    squad = dashboard_data.load_squad_predictions(1, 5, SEASON, 5)
    expected = _registered(setup).players.set_index("code")["p_start"]
    assert squad["p_start"].tolist() == [expected[1000], expected[1001]]
    assert squad["availability_status"].tolist() == ["available", "doubtful_25"]


def test_impact_is_judged_on_the_percentage_point_gap(setup):
    result = _live(setup)
    p = result.players.set_index("code")["p_start"]
    rows = result.explanation
    assert (rows["gap"] - (rows["p_start_if_nailed_on"] - rows["code"].map(p))).abs().max() < 1e-12
    assert [explanation.impact(g) for g in (0.30, 0.12, 0.03, 0.01, -0.01, -0.05)] == [
        "Holding him back a lot", "Holding him back", "Holding him back a little",
        "In line with a nailed-on starter", "In line with a nailed-on starter", "Helping him"]
