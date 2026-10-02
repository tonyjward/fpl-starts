"""The chat agent's tools (dashboard/tools.py) on the synthetic squad in
dashboard/tests/fakes.py: a 4-4-2 with Saka (40%), Gvardiol (55%) and João
Pedro (62%) at risk, a bench of Pickford, Muñoz, Rice and Isak, a £1.5m
bank and three Arsenal players."""

import copy
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")


def _load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


fakes = _load("dashboard_fakes", os.path.join(DASHBOARD_DIR, "tests", "fakes.py"))
sys.path.insert(0, DASHBOARD_DIR)  # tools.py imports squad.py as the app does
import squad as sq  # noqa: E402
import tools  # noqa: E402

from fpl_starts import refresh  # noqa: E402


def _context(transfers=None, **status_overrides):
    state = {}
    assert sq.submit_team_id(state, str(fakes.VALID_TEAM_ID), fakes.fetch_team_summary) is None
    assert sq.load_official_squad(state, fakes.universe(), fakes.LAST_COMPLETED_GW, fakes.fetch_team_picks) is None
    assert sq.submit_transfer_message(state, transfers or "No changes", fakes.universe()) is None
    state["predictions"] = fakes.predictions()
    status = fakes.status()
    for element, change in status_overrides.items():
        status[fakes.code(int(element.lstrip("e")))].update(change)
    return tools.Context(state=state, universe=fakes.universe(), status=status, data_as_of=fakes.DATA_AS_OF)


# --- explain_player -----------------------------------------------------------------------

def test_explain_player_in_squad():
    text = tools.explain_player(_context(), "Saka")
    assert "Bukayo Saka (Arsenal, MID) -- in your squad, £10.1m." in text
    assert "Chance of starting in gameweek 6: 40%. A regular starter would be at 96%" in text
    assert 'doubtful (75% chance of playing) -- FPL news: "Knock - 75% chance of playing" (added 2026-09-20)' in text
    assert text.endswith("FPL data as of 21 Sep 2026, 15:54 UTC.")


def test_explain_any_player_and_ambiguity():
    ctx = _context()
    assert "could be more than one player" in tools.explain_player(ctx, "Palmer")
    text = tools.explain_player(ctx, "Cole Palmer")
    assert "Cole Palmer (Chelsea, MID), £9.7m." in text and "Chance of starting in gameweek 6: 80%" in text
    assert "couldn't find a player called 'Zlatan Nobody'" in tools.explain_player(ctx, "Zlatan Nobody")


def test_explain_player_prefers_the_squad_player():
    assert "Bruno Fernandes (Man Utd, MID) -- in your squad" in tools.explain_player(_context(), "Fernandes")


# --- squad_risks -------------------------------------------------------------------------------

def test_squad_risks_lists_starters_below_the_threshold_with_legal_swaps():
    text = tools.squad_risks(_context())
    lines = text.splitlines()
    risks = [l for l in lines if l.startswith("- ")]
    assert [l.split(":")[0] for l in risks] == ["- MID Saka", "- DEF Gvardiol", "- FWD João Pedro"]
    assert "40% -- your vice-captain" in risks[0]
    assert "Bench options that keep a legal formation: Muñoz (91%), Rice (93%)." in risks[0]
    assert "Pickford" not in text and "Isak" not in text  # a keeper can't play outfield; Isak is less likely


def test_squad_risks_respects_the_formation():
    text = tools.squad_risks(_context(), threshold=0.99)
    raya = next(l for l in text.splitlines() if "Raya" in l)
    assert "No bench player is more likely to start in a legal formation." in raya
    starters = [p for p in _context().state["current_squad"] if p["slot"] <= 11]
    assert tools._formation_ok(starters)
    two_keepers = [dict(starters[0])] + starters[1:10] + [dict(starters[0], code=0)]
    assert not tools._formation_ok(two_keepers)


def test_squad_risks_puts_transfers_in_the_outgoing_players_slot():
    text = tools.squad_risks(_context("João Pedro out for Watkins"))
    assert "João Pedro" not in text and "Watkins" not in text  # Watkins (95%) starts, not at risk


def test_squad_risks_none():
    assert "None -- every starter is at 30% or better." in tools.squad_risks(_context(), threshold=0.30)


# --- find_replacements ---------------------------------------------------------------------------

def test_replacements_for_a_squad_player_use_his_position_and_the_estimated_budget():
    text = tools.find_replacements(_context(), replacing="Saka")
    lines = text.splitlines()
    assert lines[0].startswith("Midfielders likely to start in gameweek 6 (75%+), not in your squad, up to £11.6m")
    assert "budget £11.6m = £10.1m for Saka + £1.5m in the bank (estimate: your gameweek 5 bank" in lines[1]
    picks = [l for l in lines if l.startswith("- ")]
    assert [l.split(" (")[0] for l in picks] == ["- Martin Ødegaard", "- Harry Wilson", "- Mateus Fernandes",
                                                  "- Cole Palmer"]
    assert "Hamstring - 50% chance of playing" in picks[-1]


def test_replacements_respect_the_three_per_club_limit_and_the_budget():
    text = tools.find_replacements(_context(), replacing="Mbeumo")  # keeps all 3 Arsenal players
    assert "Ødegaard" not in text and "Left out: players from Arsenal" in text
    assert "Cole Palmer" not in text  # £9.7m > £8.0m + £1.5m


def test_bank_estimate_includes_transfers_and_a_user_override():
    ctx = _context("João Pedro out for Calvert-Lewin")  # +£7.7m -£6.0m
    assert "+ £3.2m in the bank" in tools.find_replacements(ctx, replacing="Saka")
    text = tools.find_replacements(ctx, replacing="Saka", bank=0.4)
    assert "+ £0.4m in the bank (your figure)" in text
    assert ctx.state["bank_override"] == 0.4  # remembered for later questions
    assert "(your figure)" in tools.find_replacements(ctx, replacing="Gordon")


def test_an_impossible_negative_bank_estimate_is_not_used_as_a_budget():
    ctx = _context("Gordon out for Cole Palmer")  # £1.5m + £7.5m - £9.7m < 0 at current prices
    text = tools.find_replacements(ctx, replacing="Saka")
    assert ", any price (no budget given)" in text.splitlines()[0]
    assert "would leave -£0.7m in the bank, which FPL doesn't allow" in text
    assert "£-" not in text


def test_replacements_by_position_and_price():
    text = tools.find_replacements(_context(), position="defenders", max_price=5.0, min_chance=0.6)
    assert "Defenders likely to start in gameweek 6 (60%+), not in your squad, up to £5.0m" in text
    assert "João Pedro Loureiro da Costa (West Ham, DEF), £4.5m: 70%" in text
    assert "Position must be" in tools.find_replacements(_context(), position="coach")
    assert "isn't in your squad" in tools.find_replacements(_context(), replacing="Watkins")


# --- player_news ---------------------------------------------------------------------------------

def test_squad_news_flags_changes_since_the_forecast():
    text = tools.player_news(_context())
    saka = next(l for l in text.splitlines() if "Saka" in l)
    isak = next(l for l in text.splitlines() if "Isak" in l)
    assert "This has changed since the forecast (made 20 Sep 2026, 10:00 UTC), which assumed available" in saka
    assert 'injured (0% chance of playing) -- FPL news: "Groin injury - Expected back 18 Oct"' in isak
    assert "changed" not in isak  # the forecast already assumed he was injured
    assert len([l for l in text.splitlines() if l.startswith("- ")]) == 2


def test_one_players_news_is_part_of_explain_player():
    assert "doubtful (50% chance of playing)" in tools.explain_player(_context(), "Cole Palmer")
    assert "changed" not in tools.explain_player(_context(), "Cole Palmer")  # the forecast assumed doubtful
    saka = tools.explain_player(_context(), "Saka")
    assert "This has changed since the forecast (made 20 Sep 2026, 10:00 UTC), which assumed available" in saka


def test_no_squad_news():
    quiet = _context(e9={"status": "a", "news": "", "chance_of_playing_next_round": None},
                     e15={"status": "i", "news": ""})
    assert tools.player_news(quiet).startswith("No FPL injury, suspension or news flags")


# --- refresh_data -----------------------------------------------------------------------------------

def test_refresh_reports_what_changed_for_the_squad():
    ctx = _context()
    after = _context(e9={"status": "i", "chance_of_playing_next_round": 0, "news": "Ankle injury"})
    after.state["predictions"].players.loc[after.state["predictions"].players["code"] == fakes.code(9), "p_start"] = 0.05
    after.data_as_of = "20260922T090000Z"
    result = refresh.RefreshResult(refresh.REFRESHED, "Refreshed our FPL data and updated the gameweek 6 forecast.",
                                   6, "20260922T090000Z", True, "snapshot.json")
    text = tools.refresh_data(ctx, lambda: result, lambda: after)
    assert text.startswith("Refreshed our FPL data and updated the gameweek 6 forecast.")
    assert '- Saka: FPL status now injured (0% chance of playing) -- FPL news: "Ankle injury"' in text
    assert "Chance of starting 40% -> 5%." in text
    assert text.endswith("FPL data as of 22 Sep 2026, 09:00 UTC.")


def test_refresh_refused_changes_nothing():
    ctx = _context()
    result = refresh.RefreshResult(refresh.DEADLINE_PASSED, "The gameweek 6 deadline has passed.", 6, fakes.DATA_AS_OF)
    text = tools.refresh_data(ctx, lambda: result, lambda: pytest.fail("no reload after a refused refresh"))
    assert text == "The gameweek 6 deadline has passed.\nFPL data as of 21 Sep 2026, 15:54 UTC."


def test_tools_need_a_confirmed_squad_and_a_forecast():
    ctx = _context()
    ctx.state["predictions"] = None
    assert tools.explain_player(ctx, "Saka") == "No forecast is loaded yet for the upcoming gameweek."
    fresh = tools.Context(state={}, universe=fakes.universe(), status=fakes.status(), data_as_of=fakes.DATA_AS_OF)
    assert "squad hasn't been confirmed" in tools.squad_risks(fresh)
