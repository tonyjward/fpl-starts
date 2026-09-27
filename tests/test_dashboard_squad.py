"""Team onboarding and transfer-aware squad state (dashboard/squad.py), on
synthetic FPL payloads: validated team ID, immutable official squad,
structured transfer overrides and predictions selected for the effective
current squad. No Streamlit and no network -- the AppTests for the same
flow live in dashboard/tests/."""

import copy
import importlib.util
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")


def _load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


sq = _load("dashboard_squad", os.path.join(DASHBOARD_DIR, "squad.py"))
fakes = _load("dashboard_fakes", os.path.join(DASHBOARD_DIR, "tests", "fakes.py"))

JOAO_PEDRO, CALVERT_LEWIN, BRUNO, WATKINS, COLE_PALMER, HAALAND, SAKA = (
    fakes.code(e) for e in (13, 16, 8, 19, 17, 14, 9))


@pytest.fixture
def universe():
    return sq.player_universe(fakes.bootstrap())


@pytest.fixture
def state():
    """A session with a validated team and its official squad loaded."""
    s = {}
    assert sq.submit_team_id(s, str(fakes.VALID_TEAM_ID), fakes.fetch_team_summary) is None
    assert sq.load_official_squad(s, fakes.bootstrap(), fakes.fetch_team_picks) is None
    return s


def codes(squad):
    return [p["code"] for p in squad]


def _transfer(state, text, universe):
    return sq.submit_transfer_message(state, text, universe)


# --- 1-3: team ID ------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["999", "abc", "", "0", "-5", "12.5"])
def test_invalid_team_id_does_not_advance(raw):
    s = {}
    assert sq.submit_team_id(s, raw, fakes.fetch_team_summary) == sq.INVALID_TEAM_ID
    assert sq.stage(s) == sq.NO_TEAM
    assert "team_id" not in s and not s.get("team_id_validated")


def test_unreachable_fpl_does_not_advance():
    def offline(team_id):
        raise fakes.requests.ConnectionError("offline")
    s = {}
    message = sq.submit_team_id(s, str(fakes.VALID_TEAM_ID), offline)
    assert "Couldn't reach FPL" in message and sq.stage(s) == sq.NO_TEAM


def test_valid_team_id_is_persisted():
    s = {}
    assert sq.submit_team_id(s, " 7,654,321 ", fakes.fetch_team_summary) is None
    assert s["team_id"] == fakes.VALID_TEAM_ID and s["team_id_validated"] is True
    assert s["team_name"] == "Example XI"
    assert sq.stage(s) == sq.TEAM_ID_VALID


def test_validated_team_id_is_not_requested_again():
    s = {}
    sq.submit_team_id(s, str(fakes.VALID_TEAM_ID), fakes.fetch_team_summary)
    for _ in range(3):  # a Streamlit rerun re-reads the same session state
        assert sq.stage(s) != sq.NO_TEAM


def test_official_squad_is_the_last_completed_gameweeks(state):
    assert state["last_completed_gameweek"] == fakes.LAST_COMPLETED_GW
    assert codes(state["official_squad"]) == [fakes.code(e) for e in fakes.SQUAD_ELEMENTS]
    assert sq.stage(state) == sq.OFFICIAL_SQUAD_LOADED
    assert state["current_squad"] is None  # predictions wait for the transfer step


# --- 4-9: transfers -------------------------------------------------------------------------

def test_no_changes_confirms_the_official_squad(state, universe):
    assert _transfer(state, "No changes", universe) is None
    assert sq.stage(state) == sq.CURRENT_SQUAD_READY
    assert state["transfer_overrides"] == []
    assert list(state["current_squad"]) == list(state["official_squad"])


def test_one_transfer_replaces_exactly_one_player(state, universe):
    official = copy.deepcopy(state["official_squad"])
    assert _transfer(state, "I transferred João Pedro out for Dominic Calvert-Lewin.", universe) is None
    current = codes(state["current_squad"])
    assert JOAO_PEDRO not in current and CALVERT_LEWIN in current
    assert len(current) == 15 and len(set(current)) == 15
    assert [a == b for a, b in zip(current, codes(official))].count(False) == 1
    assert current.index(CALVERT_LEWIN) == codes(official).index(JOAO_PEDRO)  # same squad slot
    assert state["transfer_overrides"] == [{
        "out_code": JOAO_PEDRO, "in_code": CALVERT_LEWIN, "out_name": "João Pedro", "in_name": "Calvert-Lewin",
        "in_player": universe[CALVERT_LEWIN]}]
    assert state["raw_transfer_messages"] == ["I transferred João Pedro out for Dominic Calvert-Lewin."]
    assert state["official_squad"] == official


def test_multiple_transfers(state, universe):
    official = copy.deepcopy(state["official_squad"])
    text = "I sold Bruno Fernandes and João Pedro and bought Cole Palmer and Watkins."
    assert _transfer(state, text, universe) is None
    current = codes(state["current_squad"])
    for gone in (BRUNO, JOAO_PEDRO):
        assert gone not in current
    for new in (COLE_PALMER, WATKINS):
        assert new in current
    assert [(o["out_code"], o["in_code"]) for o in state["transfer_overrides"]] == [
        (BRUNO, COLE_PALMER), (JOAO_PEDRO, WATKINS)]
    assert state["official_squad"] == official


def test_transfers_accumulate_across_messages(state, universe):
    assert _transfer(state, "João Pedro out for Calvert-Lewin", universe) is None
    assert _transfer(state, "Calvert-Lewin -> Watkins", universe) is None
    current = codes(state["current_squad"])
    assert WATKINS in current and CALVERT_LEWIN not in current and JOAO_PEDRO not in current
    assert len(state["raw_transfer_messages"]) == 2


@pytest.mark.parametrize("text", [
    "Joao Pedro out, Calvert Lewin in",            # accents and hyphen dropped
    "replaced joão pedro with dominic calvert-lewin",
    "OUT: João Pedro IN: Calvert-Lewin",
])
def test_name_variants_resolve(state, universe, text):
    assert _transfer(state, text, universe) is None
    assert CALVERT_LEWIN in codes(state["current_squad"])


def test_outgoing_name_prefers_the_squad_player(state, universe):
    # "Fernandes" is Mateus Fernandes' display name, but the squad has Bruno.
    assert _transfer(state, "Fernandes out for Watkins", universe) is None
    assert state["transfer_overrides"][0]["out_code"] == BRUNO


# --- 10: predictions follow the current squad -------------------------------------------------

def test_predictions_are_selected_for_the_current_squad(state, universe):
    _transfer(state, "João Pedro out for Calvert-Lewin", universe)
    selected, missing = sq.squad_predictions(fakes.predictions(), state["current_squad"])
    assert set(selected.players["code"]) == set(codes(state["current_squad"]))
    assert JOAO_PEDRO not in set(selected.players["code"]) and CALVERT_LEWIN in set(selected.players["code"])
    assert set(selected.contributions["code"]) == set(codes(state["current_squad"]))
    assert missing == []
    assert set(selected.explanation["code"]) == set(codes(state["current_squad"]))
    assert len(selected.explain(CALVERT_LEWIN)) == 2
    table = sq.squad_table(state["current_squad"], selected)
    assert table.loc[table["code"] == CALVERT_LEWIN, "transferred_in"].item() is True
    assert table["p_start"].notna().all()


def test_squad_players_without_a_prediction_are_reported(state, universe):
    _transfer(state, "No changes", universe)
    full = fakes.predictions()
    full.players = full.players[full.players["code"] != HAALAND]
    _, missing = sq.squad_predictions(full, state["current_squad"])
    assert codes(missing) == [HAALAND]


def test_chat_squad_report_uses_the_current_squad(state, universe):
    _transfer(state, "João Pedro out for Calvert-Lewin", universe)
    state["predictions"] = fakes.predictions()
    report = sq.current_squad_report(state)
    assert "Calvert-Lewin" in report and "João Pedro" not in report.split("\n", 1)[1]
    assert "- Calvert-Lewin: held back by playing time at his club (started some games; would be" in report
    assert "João Pedro -> Calvert-Lewin" in report.split("\n", 1)[0]


# --- 11-14: rejected transfers leave state untouched -------------------------------------------

def _rejected(state, universe, text, expected):
    before = copy.deepcopy(dict(state))
    message = _transfer(state, text, universe)
    assert message is not None and expected in message, message
    assert dict(state) == before


def test_ambiguous_player_is_rejected(state, universe):
    _rejected(state, universe, "João Pedro out for Palmer", "could be more than one player")


def test_ambiguous_player_lists_the_candidates(state, universe):
    message = _transfer(state, "Haaland out for Wilson", universe)
    assert "Callum Wilson" in message and "Harry Wilson" in message


def test_nonexistent_player_is_rejected(state, universe):
    _rejected(state, universe, "João Pedro out for Zlatan Nobody", "couldn't find a player")


def test_outgoing_player_not_in_squad_is_rejected(state, universe):
    _rejected(state, universe, "Watkins out for Calvert-Lewin", "isn't in your squad")


def test_incoming_player_already_in_squad_is_rejected(state, universe):
    _rejected(state, universe, "João Pedro out for Haaland", "already in your squad")


def test_duplicate_incoming_player_is_rejected(state, universe):
    _rejected(state, universe, "sold João Pedro and Isak and bought Watkins and Watkins", "already in your squad")


def test_failure_part_way_through_a_message_applies_nothing(state, universe):
    _transfer(state, "No changes", universe)
    _rejected(state, universe, "João Pedro out for Watkins; Isak out for Palmer", "could be more than one player")


def test_unreadable_or_unbalanced_messages_are_rejected(state, universe):
    _rejected(state, universe, "what a weekend", "couldn't read")
    _rejected(state, universe, "sold João Pedro and Isak and bought Watkins", "2 player(s) out but 1 in")


def test_official_squad_is_never_mutated(state, universe):
    official = copy.deepcopy(state["official_squad"])
    for text in ("João Pedro out for Palmer", "João Pedro out for Calvert-Lewin", "Isak -> Watkins"):
        _transfer(state, text, universe)
    assert state["official_squad"] == official
    sq.reset_transfers(state)
    assert state["official_squad"] == official and sq.stage(state) == sq.OFFICIAL_SQUAD_LOADED


# --- 15: changing team -----------------------------------------------------------------------------

def test_change_team_clears_all_squad_state(state, universe):
    _transfer(state, "João Pedro out for Calvert-Lewin", universe)
    state.update(predictions=fakes.predictions(), pred_player="x", chat_history=[{"role": "user"}],
                 team_id_input=str(fakes.VALID_TEAM_ID), transfer_input="x", unrelated="kept")
    sq.change_team(state)
    for key in ["team_id", "team_id_validated", "official_squad", "last_completed_gameweek", "transfer_overrides",
                "current_squad", "raw_transfer_messages", "predictions", "pred_player", "chat_history",
                "team_id_input", "transfer_input"]:
        assert key not in state, key
    assert state == {"unrelated": "kept"}
    assert sq.stage(state) == sq.NO_TEAM


def test_validating_a_new_team_does_not_carry_transfers(state, universe):
    _transfer(state, "João Pedro out for Calvert-Lewin", universe)
    assert sq.submit_team_id(state, str(fakes.VALID_TEAM_ID), fakes.fetch_team_summary) is None
    assert sq.stage(state) == sq.TEAM_ID_VALID
    assert "transfer_overrides" not in state and "current_squad" not in state


# --- parsing ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["No changes", "no transfers", "Nothing.", "none", "Same team",
                                  "I haven't made any changes"])
def test_no_change_messages(text):
    assert sq.parse_transfer_message(text) == []


@pytest.mark.parametrize("text, pairs", [
    ("I transferred João Pedro out for Dominic Calvert-Lewin.", [("João Pedro", "Dominic Calvert-Lewin")]),
    ("I sold Bruno Fernandes and João Pedro and bought Palmer and Watkins.",
     [("Bruno Fernandes", "Palmer"), ("João Pedro", "Watkins")]),
    ("Saka out, Palmer in", [("Saka", "Palmer")]),
    ("swapped B.Fernandes for Cole Palmer", [("B.Fernandes", "Cole Palmer")]),
    ("Haaland -> Isak; Saka out for Palmer", [("Haaland", "Isak"), ("Saka", "Palmer")]),
])
def test_transfer_messages_parse(text, pairs):
    assert sq.parse_transfer_message(text) == pairs
