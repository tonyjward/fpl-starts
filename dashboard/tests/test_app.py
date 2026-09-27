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
    assert sorted(table["Player"]) == sorted(fakes.universe()[fakes.code(e)]["display_name"]
                                             for e in fakes.SQUAD_ELEMENTS)
    assert {"Bukayo Saka", "Bruno Fernandes"} <= set(table["Player"])  # not FPL's web names "Saka", "B.Fernandes"
    assert fake_fpl == [("2026-27", fakes.LAST_COMPLETED_GW + 1, "registered_snapshot")]
    assert len(at.tabs[0].radio) == 0 and len(at.tabs[0].number_input) == 0  # no source/gameweek options
    captions = [c.value for c in at.tabs[0].caption]
    assert captions[0] == "FPL data as of 21 Sep 2026, 15:54 UTC."
    assert any(c.startswith("Latest forecast from logistic_availability_v1") for c in captions)

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
    assert "Dominic Calvert-Lewin" in names and not any("João Pedro" in n for n in names)
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


def test_refresh_button_reports_the_outcome(fake_fpl, monkeypatch):
    import data
    from fpl_starts import refresh

    calls = []

    def fake_refresh():
        calls.append(1)
        return refresh.RefreshResult(refresh.RECENT, "Our FPL data was refreshed at 21 Sep 15:54 UTC.", 6,
                                     fakes.DATA_AS_OF)
    monkeypatch.setattr(data, "refresh_fpl_data", fake_refresh)
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    at = _submit_transfers(at, "No changes")
    at.button(key="refresh_data").click()
    at.run()
    assert calls == [1]
    assert any("Our FPL data was refreshed at 21 Sep 15:54 UTC." in i.value for i in at.tabs[0].info)
    assert len(at.tabs[0].dataframe) >= 1  # predictions still shown


def _ready_for_chat():
    at = _submit_team_id(_start(), str(fakes.VALID_TEAM_ID))
    return _submit_transfers(at, "No changes")


def test_chat_tool_calls_run_outside_the_script_thread(fake_fpl, scripted_llm):
    """LangGraph runs tools in worker threads, where Streamlit's session state
    and caches don't work -- the tools must only use the prepared snapshot."""
    llm = scripted_llm("squad_risks")
    at = _ready_for_chat()
    at.chat_input[0].set_value("Who's at risk in my squad?").run()
    assert not at.exception
    answer = at.chat_message[-1].markdown[0].value
    assert answer.startswith("From the tool: Starting XI risks for gameweek 6")
    assert "MID Saka: 40%" in answer


def test_a_refresh_from_the_chat_updates_the_predictions_tab(fake_fpl, scripted_llm, monkeypatch):
    """The chat runs after the Predictions tab is drawn; a refresh that changes
    the forecast must still show there straight away."""
    import data
    from fpl_starts import refresh

    version = {"db": 0}
    before = fakes.predictions()
    after = fakes.predictions()
    after.players.loc[after.players["code"] == fakes.code(9), "p_start"] = 0.05  # Saka: 40% -> 5%
    monkeypatch.setattr(data, "db_version", lambda: version["db"])
    monkeypatch.setattr(data, "load_gameweek_predictions",
                        lambda season, gw, source=data.SOURCE_REGISTERED, model=None: after if version["db"] else before)

    def fake_refresh():
        version["db"] = 1  # the rebuild replaces derived.db
        return refresh.RefreshResult(refresh.REFRESHED, "Refreshed our FPL data and updated the gameweek 6 forecast.",
                                     6, "20260927T181154Z", True, "snapshot.json")
    monkeypatch.setattr(data, "refresh_fpl_data", fake_refresh)
    scripted_llm("refresh_fpl_data")
    at = _ready_for_chat()
    saka = lambda: _predictions_table(at).set_index("Player").loc["Bukayo Saka", "Chance of starting"]
    assert saka() == 40
    at.chat_input[0].set_value("That data looks out of date").run()
    assert not at.exception
    assert saka() == 5
    assert "Chance of starting 40% -> 5%" in at.chat_message[-1].markdown[0].value


def test_another_users_refresh_reaches_this_session(fake_fpl, monkeypatch):
    import data

    version = {"db": 0}
    monkeypatch.setattr(data, "db_version", lambda: version["db"])
    at = _ready_for_chat()
    assert len(fake_fpl) == 1
    version["db"] = 1  # someone else's refresh rebuilt derived.db
    at.run()
    assert len(fake_fpl) == 2  # the forecast was reloaded


def test_chat_tool_writes_reach_the_session(fake_fpl, scripted_llm):
    scripted_llm("find_replacements", {"replacing": "Saka", "bank": 2.5})
    at = _ready_for_chat()
    at.chat_input[0].set_value("Replace Saka, I have £2.5m").run()
    assert not at.exception
    assert "+ £2.5m in the bank (your figure)" in at.chat_message[-1].markdown[0].value
    assert at.session_state["bank_override"] == 2.5  # remembered for the next question
