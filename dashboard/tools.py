"""The chat agent's tools for one manager's session: explain any player,
squad risks, replacements and FPL news -- each returning plain text built
from exact data, never an estimate of its own.

Data: players, prices, status and news from derived.db (as of the latest
capture, stated in every answer); chance of starting and its explanation
from the registered forecast; the manager's squad, transfers and bank from
the session (squad.py). Framework-agnostic: every tool takes a Context.
"""

from dataclasses import dataclass
from datetime import datetime

from fpl_starts import explanation

import squad as squadlib

# FPL's squad rules (fixed by the game, not stored in derived.db): a starting
# XI has exactly 1 goalkeeper, 3-5 defenders, 2-5 midfielders, 1-3 forwards,
# and a squad at most 3 players from one club.
FORMATION = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
STARTING_SLOTS = 11
AT_RISK = 0.75  # chance of starting below this counts as a risk

POSITION_WORDS = {
    "GKP": "GKP", "GK": "GKP", "GOALKEEPER": "GKP", "KEEPER": "GKP",
    "DEF": "DEF", "DEFENDER": "DEF", "DEFENCE": "DEF", "DEFENSE": "DEF",
    "MID": "MID", "MIDFIELDER": "MID", "MIDFIELD": "MID",
    "FWD": "FWD", "FORWARD": "FWD", "STRIKER": "FWD", "ATTACKER": "FWD",
}
STATUS_TEXT = {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable",
               "n": "not available (e.g. on loan)"}
# Forecast availability level -> the FPL status letter it came from.
FORECAST_STATUS = {"available": "a", "doubtful_75": "d", "doubtful_50": "d", "doubtful_25": "d", "injured": "i",
                   "suspended": "s", "unavailable": "u", "unknown": None}


@dataclass
class Context:
    state: dict        # the session: current squad, transfers, bank, loaded forecast
    universe: dict     # {code: player} from derived.db
    status: dict       # {code: latest price/status/news} from derived.db
    data_as_of: str    # latest capture, "YYYYMMDDTHHMMSSZ"


# --- shared helpers ---------------------------------------------------------------------

def when(timestamp):
    """ "20260921T155421Z" -> "21 Sep 2026, 15:54 UTC"."""
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").strftime("%d %b %Y, %H:%M UTC")
    except (TypeError, ValueError):
        return "an unknown time"


def _as_of(ctx):
    return "FPL data as of {0}.".format(when(ctx.data_as_of))


def _position(word):
    word = str(word).strip().upper()
    return POSITION_WORDS.get(word) or POSITION_WORDS.get(word.rstrip("S"))


def _forecast(ctx):
    return ctx.state.get("predictions")


def _squad(ctx):
    return list(ctx.state.get("current_squad") or [])


def _chance(ctx, code):
    players = _forecast(ctx).players
    row = players[players["code"] == code]
    return None if row.empty else float(row["p_start"].iloc[0])


def _price(ctx, code):
    cost = (ctx.status.get(code) or {}).get("now_cost")
    return None if cost is None else cost / 10


def _money(value):
    return "£{0:.1f}m".format(value)


def _name(player):
    return squadlib.describe(player)


def _resolve(ctx, name):
    """Any player: a match in the current squad first (so "Fernandes" is the
    one you own), otherwise across every player."""
    in_squad = squadlib._matches(name, _squad(ctx))
    if len(in_squad) == 1:
        return in_squad[0]
    found = squadlib._matches(name, list(ctx.universe.values()))
    if len(found) == 1:
        return found[0]
    if len(found) > 1 or len(in_squad) > 1:
        raise squadlib.TransferError(squadlib._ambiguous(name, found or in_squad))
    raise squadlib.TransferError("I couldn't find a player called '{0}'{1}".format(
        name, squadlib._suggestions(name, ctx.universe)))


def _ready(ctx, need_squad=False):
    """A message if the session isn't ready for this tool, else None."""
    if need_squad and squadlib.stage(ctx.state) != squadlib.CURRENT_SQUAD_READY:
        return "The user's squad hasn't been confirmed yet (team ID and transfers since the last gameweek)."
    if _forecast(ctx) is None:
        return "No forecast is loaded yet for the upcoming gameweek."
    return None


def _status_line(ctx, code):
    s = ctx.status.get(code) or {}
    flag = STATUS_TEXT.get(s.get("status"), "unknown status")
    if s.get("chance_of_playing_next_round") is not None and s.get("status") != "a":
        flag += " ({0}% chance of playing)".format(s["chance_of_playing_next_round"])
    if s.get("news"):
        flag += ' -- FPL news: "{0}"{1}'.format(
            s["news"], " (added {0})".format(s["news_added"][:10]) if s.get("news_added") else "")
    return flag


def _why(ctx, code):
    return explanation.summary(_forecast(ctx).explain(code))


# --- tools ------------------------------------------------------------------------------------

def explain_player(ctx, name):
    """Any player's chance of starting, why, his price and FPL status."""
    problem = _ready(ctx)
    if problem:
        return problem
    try:
        player = _resolve(ctx, name)
    except squadlib.TransferError as exc:
        return str(exc)
    meta = _forecast(ctx).metadata
    chance = _chance(ctx, player["code"])
    owned = player["code"] in {p["code"] for p in _squad(ctx)}
    lines = ["{0}{1}, {2}.".format(_name(player), " -- in your squad" if owned else "",
                                   _money(_price(ctx, player["code"])) if _price(ctx, player["code"]) else "price unknown")]
    if chance is None:
        lines.append("No forecast for him in gameweek {0} (he wasn't in the pre-deadline player list).".format(
            meta["gameweek"]))
    else:
        lines.append("Chance of starting in gameweek {0}: {1:.0%}. A regular starter would be at {2:.0%}; {3}.".format(
            meta["gameweek"], chance, meta["reference_p_start"], _why(ctx, player["code"])))
    lines.append("FPL status: {0}.".format(_status_line(ctx, player["code"])))
    lines.append(_as_of(ctx))
    return "\n".join(lines)


def _formation_ok(starters):
    counts = {pos: 0 for pos in FORMATION}
    for p in starters:
        counts[p["position"]] += 1
    return all(lo <= counts[pos] <= hi for pos, (lo, hi) in FORMATION.items())


def squad_risks(ctx, threshold=AT_RISK):
    """Starters below `threshold`, and legal swaps with more likely bench starters."""
    problem = _ready(ctx, need_squad=True)
    if problem:
        return problem
    squad = sorted(_squad(ctx), key=lambda p: p["slot"])
    starters = [p for p in squad if p["slot"] <= STARTING_SLOTS]
    bench = [p for p in squad if p["slot"] > STARTING_SLOTS]
    chance = {p["code"]: _chance(ctx, p["code"]) for p in squad}
    meta = _forecast(ctx).metadata
    lines = ["Starting XI risks for gameweek {0} (starting XI as picked for gameweek {1}, with transfers in the "
             "place of the player they replaced; a risk is a chance of starting under {2:.0%}):".format(
                 meta["gameweek"], ctx.state.get("last_completed_gameweek"), threshold)]
    at_risk = sorted((p for p in starters if chance[p["code"]] is not None and chance[p["code"]] < threshold),
                     key=lambda p: chance[p["code"]])
    unknown = [p for p in starters if chance[p["code"]] is None]
    if not at_risk:
        lines.append("None -- every starter is at {0:.0%} or better.".format(threshold))
    for p in at_risk:
        swaps = []
        for b in bench:
            if chance[b["code"]] is None or chance[b["code"]] <= chance[p["code"]]:
                continue
            lineup = [b if s["code"] == p["code"] else s for s in starters]
            if _formation_ok(lineup):
                swaps.append("{0} ({1:.0%})".format(b["web_name"], chance[b["code"]]))
        role = " -- your captain" if p.get("is_captain") else (" -- your vice-captain" if p.get("is_vice_captain") else "")
        lines.append("- {0} {1}: {2:.0%}{3}; {4}. FPL status: {5}. {6}".format(
            p["position"], p["web_name"], chance[p["code"]], role, _why(ctx, p["code"]), _status_line(ctx, p["code"]),
            "Bench options that keep a legal formation: {0}.".format(", ".join(swaps)) if swaps
            else "No bench player is more likely to start in a legal formation."))
    for p in unknown:
        lines.append("- {0} {1}: no forecast (not in the pre-deadline player list).".format(p["position"], p["web_name"]))
    lines.append(_as_of(ctx))
    return "\n".join(lines)


def _bank_estimate(ctx):
    """(bank £m, note) -- the user's own figure if given, else the last
    completed gameweek's bank adjusted for transfers at current prices."""
    if ctx.state.get("bank_override") is not None:
        return ctx.state["bank_override"], "your figure"
    bank = ctx.state.get("bank")
    if bank is None:
        return None, "unknown"
    total = bank / 10
    for o in ctx.state.get("transfer_overrides") or []:
        total += (_price(ctx, o["out_code"]) or 0) - (_price(ctx, o["in_code"]) or 0)
    if total < 0:
        # FPL never allows a negative bank, so current prices can't be what
        # the user paid and sold for -- don't budget with a wrong figure.
        return None, "unknown: at current prices your transfers since gameweek {0} would leave {1} in the bank, " \
                     "which FPL doesn't allow, so your real selling prices must differ -- tell me your bank".format(
                         ctx.state.get("last_completed_gameweek"), "-£{0:.1f}m".format(-total))
    note = "estimate: your gameweek {0} bank{1} -- FPL doesn't publish selling prices, so tell me your real bank " \
           "if it differs".format(ctx.state.get("last_completed_gameweek"),
                                  " adjusted for your transfers at current prices" if ctx.state.get("transfer_overrides")
                                  else "")
    return total, note


def find_replacements(ctx, replacing=None, position=None, max_price=None, bank=None, min_chance=AT_RISK, limit=8):
    """Players likely to start, not in the squad, by position and budget."""
    problem = _ready(ctx, need_squad=True)
    if problem:
        return problem
    if bank is not None:
        ctx.state["bank_override"] = float(bank)
    squad = _squad(ctx)
    out_player = None
    if replacing:
        try:
            out_player = squadlib.resolve_outgoing(replacing, squad, ctx.universe)
        except squadlib.TransferError as exc:
            return str(exc)
        position = out_player["position"]
    elif position:
        position = _position(position)
        if position is None:
            return "Position must be a goalkeeper, defender, midfielder or forward."

    budget_note = None
    if max_price is None and out_player is not None:
        bank_now, note = _bank_estimate(ctx)
        if bank_now is None:
            budget_note = "no budget applied -- bank {0}".format(note)
        elif _price(ctx, out_player["code"]) is not None:
            max_price = round(bank_now + _price(ctx, out_player["code"]), 1)
            budget_note = "budget {0} = {1} for {2} + {3} in the bank ({4})".format(
                _money(max_price), _money(_price(ctx, out_player["code"])), out_player["web_name"],
                _money(bank_now), note)

    per_club = {}
    for p in squad:
        if out_player is None or p["code"] != out_player["code"]:
            per_club[p["team"]] = per_club.get(p["team"], 0) + 1
    owned = {p["code"] for p in squad}
    forecast = _forecast(ctx).players.set_index("code")["p_start"]
    found, full_clubs = [], set()
    for code, p in ctx.universe.items():
        if code in owned or code not in forecast.index or (position and p["position"] != position):
            continue
        chance, price = float(forecast[code]), _price(ctx, code)
        if chance < min_chance or price is None or (max_price is not None and price > max_price + 1e-9):
            continue
        if per_club.get(p["team"], 0) >= MAX_PER_CLUB:
            full_clubs.add(p["team"])
            continue
        found.append((chance, price, p))
    found.sort(key=lambda t: (-t[0], t[1]))

    what = "{0}s".format({"GKP": "goalkeeper", "DEF": "defender", "MID": "midfielder", "FWD": "forward"}[position]) \
        if position else "players"
    lines = ["{0} likely to start in gameweek {1} ({2:.0%}+), not in your squad{3}{4}:".format(
        what.capitalize(), _forecast(ctx).metadata["gameweek"], min_chance,
        ", up to {0}".format(_money(max_price)) if max_price is not None else ", any price (no budget given)",
        " -- replacing {0}".format(out_player["web_name"]) if out_player else "")]
    if budget_note:
        lines.append("({0})".format(budget_note))
    for chance, price, p in found[:limit]:
        s = ctx.status.get(p["code"]) or {}
        flag = "" if s.get("status") == "a" else "; FPL: {0}".format(_status_line(ctx, p["code"]))
        lines.append("- {0}, {1}: {2:.0%} chance of starting{3}".format(_name(p), _money(price), chance, flag))
    if not found:
        lines.append("None found -- try a higher budget or a lower chance of starting.")
    if full_clubs:
        lines.append("Left out: players from {0} (you'd have more than {1} from the club).".format(
            ", ".join(sorted(full_clubs)), MAX_PER_CLUB))
    lines.append(_as_of(ctx))
    return "\n".join(lines)


def player_news(ctx, name=None):
    """FPL's status and news for one player, or every squad player with news
    or a status that has changed since the forecast was made."""
    problem = _ready(ctx, need_squad=name is None)
    if problem:
        return problem
    forecast = _forecast(ctx)
    assumed = forecast.players.set_index("code")["availability_status"]
    made = when(forecast.metadata.get("predicted_at"))

    def changed(code):
        if code not in assumed.index:
            return False
        return FORECAST_STATUS.get(assumed[code]) != (ctx.status.get(code) or {}).get("status")

    if name:
        try:
            players = [_resolve(ctx, name)]
        except squadlib.TransferError as exc:
            return str(exc)
    else:
        players = [p for p in _squad(ctx) if (ctx.status.get(p["code"]) or {}).get("news") or changed(p["code"])]
    lines = []
    for p in players:
        line = "- {0}: {1}.".format(_name(p), _status_line(ctx, p["code"]))
        if changed(p["code"]):
            line += " This has changed since the forecast (made {0}), which assumed {1}; refreshing the forecast " \
                    "would take it into account.".format(made, assumed[p["code"]].replace("_", " "))
        lines.append(line)
    if not lines:
        lines.append("No FPL injury, suspension or news flags for anyone in your squad.")
    lines.append(_as_of(ctx))
    return "\n".join(lines)


def refresh_data(ctx, run_refresh, reload_context):
    """Refresh our FPL data (and the forecast, before the deadline) via
    `run_refresh()` -> fpl_starts.refresh.RefreshResult, then report what
    changed for the squad. `reload_context()` returns a Context on the
    refreshed data (and reloads the session's forecast)."""
    squad = _squad(ctx)
    before_status = {p["code"]: dict(ctx.status.get(p["code"]) or {}) for p in squad}
    before_chance = {p["code"]: _chance(ctx, p["code"]) for p in squad} if _forecast(ctx) is not None else {}
    result = run_refresh()
    lines = [result.message]
    if result.status != "refreshed":
        lines.append(_as_of(ctx))
        return "\n".join(lines)
    after = reload_context()
    changes = []
    for p in squad:
        old, new = before_status[p["code"]], after.status.get(p["code"]) or {}
        news_changed = (old.get("status"), old.get("news")) != (new.get("status"), new.get("news"))
        old_chance = before_chance.get(p["code"])
        new_chance = _chance(after, p["code"]) if _forecast(after) is not None else None
        chance_changed = old_chance is not None and new_chance is not None and abs(new_chance - old_chance) >= 0.005
        if news_changed or chance_changed:
            line = "- {0}: FPL status now {1}.".format(p["web_name"], _status_line(after, p["code"]))
            if chance_changed:
                line += " Chance of starting {0:.0%} -> {1:.0%}.".format(old_chance, new_chance)
            changes.append(line)
    lines.append("Changes for your squad:" if changes else "Nothing has changed for anyone in your squad.")
    lines.extend(changes)
    lines.append(_as_of(after))
    return "\n".join(lines)
