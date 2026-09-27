"""Team onboarding and transfer-aware squad state for the dashboard.

The user's squad is established in stages before any squad prediction is
shown:

    NO_TEAM -> TEAM_ID_VALID -> OFFICIAL_SQUAD_LOADED
            -> TRANSFER_STATE_CONFIRMED -> CURRENT_SQUAD_READY

- the team ID is checked against the FPL API (it must exist);
- the official squad is the team as it stood at the end of the last
  completed gameweek -- stored once and never mutated;
- transfers made since then are told in plain English, parsed into
  structured (out, in) overrides on stable player codes, and applied on top:

      official_squad + transfer_overrides = current_squad

Framework-agnostic: every function takes `state`, any mutable mapping --
`st.session_state` in the app, a plain dict in tests. Predictions come from
`fpl_starts.pstart` only; this module just selects the current squad's rows.
"""

import difflib
import re
import unicodedata

import requests

from fpl_starts import explanation, pstart

NO_TEAM = "NO_TEAM"
TEAM_ID_VALID = "TEAM_ID_VALID"
OFFICIAL_SQUAD_LOADED = "OFFICIAL_SQUAD_LOADED"
TRANSFER_STATE_CONFIRMED = "TRANSFER_STATE_CONFIRMED"
CURRENT_SQUAD_READY = "CURRENT_SQUAD_READY"

# Everything specific to one FPL team; `change_team` clears all of it.
SQUAD_STATE_KEYS = [
    "team_id", "team_id_validated", "team_name", "manager_name", "official_squad", "last_completed_gameweek",
    "transfer_overrides", "raw_transfer_messages", "transfer_state_confirmed", "current_squad",
    "predictions", "pred_player", "chat_history",
    # widget values, so nothing typed for one team is shown for the next
    "team_id_input", "transfer_input",
]

INVALID_TEAM_ID = "That team ID doesn't appear to be valid. Please check it and try again."
FRESHNESS_MESSAGE = ("I only know your team as it stood at the end of the last completed gameweek. "
                     "If you've made any transfers since then, tell me what you've changed.")


class TransferError(ValueError):
    """A transfer message that can't be applied as stated; state is unchanged."""


# --- stages -------------------------------------------------------------------------

def stage(state):
    if not state.get("team_id_validated"):
        return NO_TEAM
    if state.get("official_squad") is None:
        return TEAM_ID_VALID
    if not state.get("transfer_state_confirmed"):
        return OFFICIAL_SQUAD_LOADED
    if state.get("current_squad") is None:
        return TRANSFER_STATE_CONFIRMED
    return CURRENT_SQUAD_READY


def change_team(state):
    for key in SQUAD_STATE_KEYS:
        state.pop(key, None)


def submit_team_id(state, raw, fetch_team_summary):
    """Validate `raw` against the FPL API and, if the team exists, start a
    fresh session for it. Returns None on success, else a message to show
    (state unchanged)."""
    text = re.sub(r"[\s,]", "", str(raw or ""))
    if not text.isdigit() or int(text) <= 0:
        return INVALID_TEAM_ID
    team_id = int(text)
    try:
        summary = fetch_team_summary(team_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return INVALID_TEAM_ID
        return "Couldn't check that team ID with FPL right now ({0}). Please try again.".format(exc)
    except requests.RequestException as exc:
        return "Couldn't reach FPL to check that team ID ({0}). Please try again.".format(exc)
    change_team(state)
    state["team_id"] = team_id
    state["team_id_validated"] = True
    state["team_name"] = summary.get("name")
    state["manager_name"] = " ".join(p for p in (summary.get("player_first_name"),
                                                 summary.get("player_last_name")) if p) or None
    return None


# --- players ----------------------------------------------------------------------------

def normalise(name):
    """Lowercase, accents folded, punctuation as spaces: "B.Fernandes" ->
    "b fernandes", "Calvert-Lewin" -> "calvert lewin", "João" -> "joao"."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded.lower()).split())


def _keys(player):
    full, web, second = normalise(player["full_name"]), normalise(player["web_name"]), normalise(player["second_name"])
    first = full.split()[0] if full else ""
    exact = {web, full, normalise(player["known_name"])} - {""}
    near = {second, "{0} {1}".format(first, second.split()[-1] if second else "").strip(), web.split()[-1] if web else ""}
    tokens = set(full.split()) | set(web.split()) | set(normalise(player["known_name"]).split())
    return exact, near - {""}, tokens


def _matches(query, players):
    """Players matching `query` at the first tier with any match: exact web/
    full/known name, then surname or "first last", then all query words."""
    q = normalise(query)
    if not q:
        return []
    keyed = [(p, _keys(p)) for p in players]
    for tier in range(3):
        if tier < 2:
            found = [p for p, k in keyed if q in k[tier]]
        else:
            found = [p for p, k in keyed if set(q.split()) <= k[2]]
        if found:
            return found
    return []


def describe(player):
    return "{0} ({1}, {2})".format(player["full_name"], player["team"], player["position"])


def _suggestions(query, universe):
    names = {}
    for p in universe.values():
        names.setdefault(normalise(p["web_name"]), p)
        names.setdefault(normalise(p["full_name"]), p)
    close = difflib.get_close_matches(normalise(query), list(names), n=3, cutoff=0.75)
    unique = {names[c]["code"]: names[c] for c in close}
    return "; did you mean {0}?".format(" or ".join(describe(p) for p in unique.values())) if unique else ""


def _ambiguous(query, found):
    listed = ", ".join(describe(p) for p in found[:5])
    return "'{0}' could be more than one player: {1}. Which one did you mean?".format(query, listed)


def resolve_outgoing(query, squad, universe):
    found = _matches(query, squad)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise TransferError(_ambiguous(query, found))
    elsewhere = _matches(query, list(universe.values()))
    if elsewhere:
        raise TransferError("'{0}' isn't in your squad, so they can't be transferred out.".format(query))
    raise TransferError("I couldn't find a player called '{0}'{1}".format(query, _suggestions(query, universe)))


def resolve_incoming(query, squad, universe):
    found = _matches(query, list(universe.values()))
    if not found:
        raise TransferError("I couldn't find a player called '{0}'{1}".format(query, _suggestions(query, universe)))
    if len(found) > 1:
        raise TransferError(_ambiguous(query, found))
    player = found[0]
    if any(p["code"] == player["code"] for p in squad):
        raise TransferError("{0} is already in your squad.".format(describe(player)))
    return player


# --- transfer messages ---------------------------------------------------------------------

_NO_CHANGES = re.compile(
    r"(no|nope|none|nothing|n/?a|same|same (team|squad)|unchanged|no (changes?|transfers?)( (made|at all))?"
    r"|(i )?(haven'?t|have not|didn'?t|did not) (made|make|done|do) any (changes|transfers))[.! ]*", re.I)
_CLAUSES = re.compile(r";|\n|\.\s+|\bthen\b|\balso\b", re.I)
_OUT = r"sold|sell|removed|dropped|took out|taken out|transferred out|got rid of|offloaded|benched out"
_IN = r"bought|buy|brought in|signed|got|picked up|added|replaced (?:him|her|them) with"
_PATTERNS = [
    re.compile(r"^(?:replaced|swapped|switched|changed|subbed|transferred|traded|moved)\s+(?P<out>.+?)"
               r"(?:\s+out)?\s+(?:with|for|to)\s+(?P<in>.+?)(?:\s+in)?$", re.I),
    re.compile(r"^(?:{0})\s+(?P<out>.+?)\s*,?\s+(?:and\s+)?(?:{1}|for)\s+(?P<in>.+)$".format(_OUT, _IN), re.I),
    re.compile(r"^out\s*:?\s*(?P<out>.+?)\s*,?\s+in\s*:?\s*(?P<in>.+)$", re.I),
    re.compile(r"^in\s*:?\s*(?P<in>.+?)\s*,?\s+out\s*:?\s*(?P<out>.+)$", re.I),
    re.compile(r"^(?P<out>.+?)\s*(?:->|=>|→)\s*(?P<in>.+)$", re.I),
    re.compile(r"^(?P<out>.+?)\s+out\s*,?\s+(?:and\s+)?(?:for\s+|with\s+)?(?P<in>.+?)(?:\s+in)?$", re.I),
]
_FILLER = re.compile(r"^(?:i've|i have|i just|i also|we've|okay|yeah|also|and|yes|so|ok|we|i)\b[\s,]*", re.I)


def _split_names(phrase):
    names = [n.strip(" .!?'\"") for n in re.split(r",|\band\b|&|\+", phrase, flags=re.I)]
    names = [re.sub(r"^(?:the|player)\s+", "", n, flags=re.I) for n in names]
    return [n for n in names if n]


def parse_transfer_message(text):
    """[] for "no changes", else [(out_name, in_name), ...] in the order
    given. Raises TransferError if the message can't be read as transfers."""
    t = " ".join(str(text or "").split())
    if not t:
        raise TransferError("Tell me what you've changed, or say \"no changes\".")
    if _NO_CHANGES.fullmatch(t):
        return []
    pairs = []
    for clause in filter(None, (c.strip(" ,") for c in _CLAUSES.split(t))):
        while True:  # drop leading filler words ("and I've ...")
            stripped = _FILLER.sub("", clause, count=1)
            if stripped == clause:
                break
            clause = stripped
        for pattern in _PATTERNS:
            m = pattern.match(clause)
            if m:
                outs, ins = _split_names(m.group("out")), _split_names(m.group("in"))
                break
        else:
            raise TransferError("I couldn't read \"{0}\" as a transfer. Try something like \"Player A out for "
                                "Player B\", or say \"no changes\".".format(clause))
        if len(outs) != len(ins) or not outs:
            raise TransferError("\"{0}\" has {1} player(s) out but {2} in -- each transfer swaps one player "
                                "for another.".format(clause, len(outs), len(ins)))
        pairs.extend(zip(outs, ins))
    return pairs


def derive_current_squad(official_squad, overrides):
    """A new squad: the official squad with each override's outgoing player
    replaced, in the same slot, by the incoming player."""
    squad = [dict(p) for p in official_squad]
    for o in overrides:
        index = next(i for i, p in enumerate(squad) if p["code"] == o["out_code"])
        incoming = dict(o["in_player"])
        incoming.update(slot=squad[index]["slot"], multiplier=None, is_captain=False, is_vice_captain=False,
                        transferred_in=True)
        squad[index] = incoming
    return tuple(squad)


def submit_transfer_message(state, text, universe):
    """Parse and apply one message about transfers since the last completed
    gameweek. Returns None on success, else a message to show -- a message
    that fails anywhere changes nothing."""
    try:
        pairs = parse_transfer_message(text)
        overrides = list(state.get("transfer_overrides") or [])
        squad = list(derive_current_squad(state["official_squad"], overrides))
        for out_name, in_name in pairs:
            out_player = resolve_outgoing(out_name, squad, universe)
            in_player = resolve_incoming(in_name, squad, universe)
            override = {"out_code": out_player["code"], "in_code": in_player["code"],
                        "out_name": out_player["web_name"], "in_name": in_player["web_name"],
                        "in_player": in_player}
            overrides.append(override)
            squad = list(derive_current_squad(state["official_squad"], overrides))
    except TransferError as exc:
        return str(exc)
    state["transfer_overrides"] = overrides
    state["raw_transfer_messages"] = list(state.get("raw_transfer_messages") or []) + [str(text)]
    state["transfer_state_confirmed"] = True
    state["current_squad"] = tuple(squad)
    return None


def reset_transfers(state):
    state["transfer_overrides"] = []
    state["raw_transfer_messages"] = []
    state["transfer_state_confirmed"] = False
    state["current_squad"] = None
    for key in ("predictions", "pred_player"):
        state.pop(key, None)


def load_official_squad(state, universe, gw, fetch_team_picks):
    """The validated team as it stood at the end of gameweek `gw` (the last
    completed one): the picks come from the FPL API, the players from
    `universe` ({code: player}, from derived.db). Returns None on success,
    else a message to show."""
    if gw is None:
        return "No gameweek has finished yet this season, so there's no official squad to start from."
    try:
        picks = fetch_team_picks(state["team_id"], gw)["picks"]
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "Team {0} has no squad for gameweek {1} (it may have been created after it).".format(
                state["team_id"], gw)
        return "Couldn't load the squad from FPL right now ({0}).".format(exc)
    except requests.RequestException as exc:
        return "Couldn't reach FPL to load the squad ({0}).".format(exc)
    by_element = {p["element"]: p for p in universe.values()}
    unknown = [pick["element"] for pick in picks if pick["element"] not in by_element]
    if unknown:
        return ("Your squad includes {0} player(s) our FPL data doesn't have yet (FPL id {1}) -- "
                "our data needs refreshing.".format(len(unknown), ", ".join(map(str, unknown))))
    squad = []
    for pick in sorted(picks, key=lambda p: p["position"]):
        player = dict(by_element[pick["element"]])
        player.update(slot=pick["position"], multiplier=pick["multiplier"], is_captain=pick["is_captain"],
                      is_vice_captain=pick["is_vice_captain"], transferred_in=False)
        squad.append(player)
    state["official_squad"] = tuple(squad)
    state["last_completed_gameweek"] = gw
    reset_transfers(state)
    return None


# --- predictions for the current squad ------------------------------------------------------------

def squad_predictions(predictions, squad):
    """`predictions` (a pstart.PStartPredictions) cut down to `squad`'s
    players, and the squad players it has no prediction for."""
    codes = [p["code"] for p in squad]
    players = predictions.players[predictions.players["code"].isin(codes)]
    contributions = predictions.contributions[predictions.contributions["code"].isin(codes)]
    explained = predictions.explanation[predictions.explanation["code"].isin(codes)]
    missing = [p for p in squad if p["code"] not in set(players["code"])]
    return pstart.PStartPredictions(players.reset_index(drop=True), contributions.reset_index(drop=True),
                                    dict(predictions.metadata), explained.reset_index(drop=True)), missing


def squad_table(squad, predictions):
    """The squad in slot order with each player's P(start) columns (NaN where
    the predictions have no row for him)."""
    import pandas as pd

    base = pd.DataFrame([{k: p.get(k) for k in ("slot", "code", "web_name", "team", "position", "multiplier",
                                                 "is_captain", "is_vice_captain", "transferred_in")}
                         for p in squad])
    columns = ["code", "p_start", "availability_status", "last_gw_role", "current_season_start_rate",
               "previous_season_start_rate"]
    return base.merge(predictions.players[columns], on="code", how="left").sort_values("slot").reset_index(drop=True)


def current_squad_report(state):
    """Plain-text current squad with each player's P(start) and what is
    holding him back compared with a nailed-on starter -- what the chat
    agent's squad tool returns."""
    if stage(state) != CURRENT_SQUAD_READY:
        return "The user's squad hasn't been confirmed yet (team ID and transfers since the last gameweek)."
    if state.get("predictions") is None:
        return "No predictions are loaded for the current squad (see the Predictions tab)."
    predictions, missing = squad_predictions(state["predictions"], state["current_squad"])
    table = squad_table(state["current_squad"], predictions)
    meta = predictions.metadata
    lines = ["{0} P(start) for GW{1} ({2}), current squad after transfers since GW{3}: {4}".format(
        meta["model_id"], meta["gameweek"], meta["source"], state["last_completed_gameweek"],
        "; ".join("{0} -> {1}".format(o["out_name"], o["in_name"]) for o in state["transfer_overrides"]) or "none")]
    lines.append(table.drop(columns=["code"]).round(3).to_string(index=False))
    lines.append("Why, compared with a nailed-on starter ({0}; {1:.0%}):".format(
        meta["reference_description"], meta["reference_p_start"]))
    for p in state["current_squad"]:
        if p["code"] in set(predictions.players["code"]):
            lines.append("- {0}: {1}".format(p["web_name"], explanation.summary(predictions.explain(p["code"]))))
    if missing:
        lines.append("No prediction for: " + ", ".join(describe(p) for p in missing))
    return "\n".join(lines)
