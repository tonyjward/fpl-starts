"""Onboarding flow through the real Streamlit app (AppTest, no browser)."""

import os

from streamlit.testing.v1 import AppTest

import fakes
import squad

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _start():
    return AppTest.from_file(APP, default_timeout=60).run()


def _submit_team_id(at, value):
    at.text_input(key="team_id_input").input(value)
    at.button[0].click()  # the form's submit button
    return at.run()


def _submit_transfers(at, text):
    at.text_input(key="transfer_input").input(text)
    next(b for b in at.button if b.label in ("Confirm squad", "Update squad")).click()
    return at.run()


def _predictions_table(at):
    return at.tabs[0].dataframe[0].value


def test_team_id_is_the_first_interaction(fake_fpl):
    at = _start()
    box = at.text_input(key="team_id_input")
    assert box.label == "What is your FPL team ID?"
    assert box.placeholder == "e.g. 1234567" and box.value == ""
    assert any("Where do I find my team ID?" in c.value for c in at.caption)
    assert len(at.tabs) == 0 and not fake_fpl


def test_invalid_team_id_stays_on_the_team_id_step(fake_fpl):
    at = _submit_team_id(_start(), "999")
    assert [e.value for e in at.error] == [squad.INVALID_TEAM_ID]
    assert at.text_input(key="team_id_input") is not None
    assert "team_id_validated" not in at.session_state
    assert "official_squad" not in at.session_state
    assert len(at.tabs) == 0 and not fake_fpl
    assert not any(w.key == "transfer_input" for w in at.text_input)


def test_happy_path_no_transfers(fake_fpl):
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    assert at.session_state["team_id"] == fakes.VALID_TEAM_ID
    assert at.session_state["last_completed_gameweek"] == fakes.LAST_COMPLETED_GW
    assert squad.FRESHNESS_MESSAGE in [i.value for i in at.info]
    assert not any(w.key == "team_id_input" for w in at.text_input)
    assert len(at.tabs) == 0 and not fake_fpl  # no predictions before the transfer step

    at = _submit_transfers(at, "No changes")
    assert at.session_state["transfer_state_confirmed"] is True
    table = _predictions_table(at)
    assert sorted(table["Player"]) == sorted(p[1] for p in fakes.PLAYERS[:15])
    assert fake_fpl == [("2026-27", fakes.LAST_COMPLETED_GW + 1, "registered_snapshot")]
    assert len(at.tabs[0].radio) == 0 and len(at.tabs[0].number_input) == 0  # no source/gameweek options
    assert "Latest forecast from logistic_availability_v1" in at.tabs[0].caption[0].value

    breakdown = at.tabs[0].dataframe[1].value  # the selected player's grouped explanation
    assert list(breakdown.columns) == ["Factor", "What we know", "Impact", "Chance without this issue"]
    assert at.tabs[0].markdown[0].value.startswith("**Chance of starting:")
    assert any("A regular starter" in c.value for c in at.tabs[0].caption)

    at.run()  # a plain rerun keeps the validated team
    assert not any(w.key == "team_id_input" for w in at.text_input)
    assert [t.label for t in at.tabs] == ["Predictions", "Ask the agent"]


def test_transfer_changes_every_prediction_view(fake_fpl):
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    at = _submit_transfers(at, "I transferred João Pedro out for Dominic Calvert-Lewin.")
    names = set(_predictions_table(at)["Player"])
    assert "Calvert-Lewin" in names and "João Pedro" not in names
    assert all("João Pedro" not in option for option in at.selectbox(key="pred_player").options)


def test_rejected_transfer_keeps_the_squad_step(fake_fpl):
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    at = _submit_transfers(at, "João Pedro out for Palmer")
    assert any("could be more than one player" in w.value for w in at.warning)
    assert len(at.tabs) == 0 and "transfer_overrides" in at.session_state
    assert at.session_state["transfer_overrides"] == []


def test_change_team_returns_to_the_team_id_step(fake_fpl):
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    at = _submit_transfers(at, "João Pedro out for Calvert-Lewin")
    at.button(key="change_team").click()
    at.run()
    box = at.text_input(key="team_id_input")
    assert box.value == "" and len(at.tabs) == 0
    for key in ("team_id", "official_squad", "transfer_overrides", "current_squad", "predictions"):
        assert key not in at.session_state
