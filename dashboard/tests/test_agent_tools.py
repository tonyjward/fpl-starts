"""The agent's session tools are thin wrappers: each LangChain tool calls the
matching tools.py function on the session's current context. No LLM."""

import agent
import fakes
import squad
import tools


def _context():
    state = {}
    squad.submit_team_id(state, str(fakes.VALID_TEAM_ID), fakes.fetch_team_summary)
    squad.load_official_squad(state, fakes.universe(), fakes.LAST_COMPLETED_GW, fakes.fetch_team_picks)
    squad.submit_transfer_message(state, "No changes", fakes.universe())
    state["predictions"] = fakes.predictions()
    return tools.Context(state=state, universe=fakes.universe(), status=fakes.status(), data_as_of=fakes.DATA_AS_OF)


def test_session_tools_call_the_tools_module_on_the_current_context():
    ctx = _context()
    refreshed = []
    app_tools = {t.name: t for t in agent.make_app_tools(lambda: "squad report", lambda: ctx,
                                                          lambda: refreshed.append(1) or "refreshed")}
    assert sorted(app_tools) == ["explain_player", "find_replacements", "get_my_current_squad_predictions",
                                 "player_news", "refresh_fpl_data", "squad_risks"]
    assert app_tools["get_my_current_squad_predictions"].invoke({}) == "squad report"
    assert app_tools["explain_player"].invoke({"name": "Saka"}) == tools.explain_player(ctx, "Saka")
    assert app_tools["squad_risks"].invoke({}) == tools.squad_risks(ctx)
    assert app_tools["find_replacements"].invoke({"replacing": "Saka"}) == tools.find_replacements(ctx, replacing="Saka")
    assert app_tools["player_news"].invoke({}) == tools.player_news(ctx)
    assert app_tools["refresh_fpl_data"].invoke({}) == "refreshed" and refreshed == [1]


def test_tool_descriptions_tell_the_model_when_to_use_them():
    app_tools = {t.name: t for t in agent.make_app_tools(lambda: "", lambda: None, lambda: "")}
    assert "legal formation" in app_tools["squad_risks"].description
    assert "3-per-club" in app_tools["find_replacements"].description
    assert "deadline" in app_tools["refresh_fpl_data"].description
    assert "who'll score" in agent.SYSTEM_PROMPT  # it declines points/captaincy questions
