"""Synthetic FPL API payloads and P(start) predictions for the dashboard
tests -- invented data shaped like bootstrap-static, entry/ and picks/,
never real API responses. Shared by the root suite's squad-logic tests
(tests/test_dashboard_squad.py) and the Streamlit AppTests here."""

import math

import pandas as pd
import requests

from fpl_starts import pstart
from fpl_starts.ml import spec

VALID_TEAM_ID = 7654321
LAST_COMPLETED_GW = 5

# element id, web_name, first_name, second_name, team id, element_type
PLAYERS = [
    (1, "Raya", "David", "Raya Martín", 1, 1),
    (2, "Pickford", "Jordan", "Pickford", 3, 1),
    (3, "Gabriel", "Gabriel", "dos Santos Magalhães", 1, 2),
    (4, "Saliba", "William", "Saliba", 2, 2),
    (5, "Virgil", "Virgil", "van Dijk", 4, 2),
    (6, "Gvardiol", "Joško", "Gvardiol", 5, 2),
    (7, "Muñoz", "Daniel", "Muñoz Mejía", 2, 2),
    (8, "B.Fernandes", "Bruno", "Borges Fernandes", 6, 3),
    (9, "Saka", "Bukayo", "Saka", 1, 3),
    (10, "Mbeumo", "Bryan", "Mbeumo", 6, 3),
    (11, "Gordon", "Anthony", "Gordon", 7, 3),
    (12, "Rice", "Declan", "Rice", 12, 3),
    (13, "João Pedro", "João Pedro", "Junqueira de Jesus", 8, 4),
    (14, "Haaland", "Erling", "Haaland", 5, 4),
    (15, "Isak", "Alexander", "Isak", 4, 4),
    # not in the official squad
    (16, "Calvert-Lewin", "Dominic", "Calvert-Lewin", 9, 4),
    (17, "Palmer", "Cole", "Palmer", 8, 3),
    (18, "Palmer", "Alex", "Palmer", 10, 1),
    (19, "Watkins", "Ollie", "Watkins", 11, 4),
    (20, "Costinha", "João Pedro", "Loureiro da Costa", 12, 2),
    (21, "Fernandes", "Mateus", "Fernandes", 7, 3),
    (22, "Wilson", "Callum", "Wilson", 12, 4),
    (23, "Wilson", "Harry", "Wilson", 13, 3),
    (24, "Ødegaard", "Martin", "Ødegaard", 1, 3),
]
TEAMS = ["Arsenal", "Crystal Palace", "Everton", "Liverpool", "Man City", "Man Utd", "Newcastle", "Chelsea",
         "Leeds", "Ipswich", "Aston Villa", "West Ham", "Fulham"]
# A legal 4-4-2 XI (slots 1-11), then the bench: GK, DEF, MID, FWD. At most
# 3 players per club (3 from Arsenal).
SQUAD_ELEMENTS = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 2, 7, 12, 15]
CAPTAIN, VICE_CAPTAIN = 14, 9
# FPL's known_name, set only for some players (shown instead of the full name).
KNOWN_NAMES = {8: "Bruno Fernandes", 13: "João Pedro"}
BANK = 15  # £1.5m, in FPL's tenths
DATA_AS_OF = "20260921T155421Z"

# Chance of starting in the fake forecast.
CHANCE = {1: .95, 2: .94, 3: .93, 4: .90, 5: .96, 6: .55, 7: .91, 8: .90, 9: .40, 10: .88, 11: .85, 12: .93,
          13: .62, 14: .97, 15: .20, 16: .90, 17: .80, 18: .05, 19: .95, 20: .70, 21: .85, 22: .30, 23: .88,
          24: .90}
# Price, in FPL's tenths of £1m.
PRICE = {1: 55, 2: 50, 3: 60, 4: 60, 5: 62, 6: 58, 7: 50, 8: 85, 9: 101, 10: 80, 11: 75, 12: 65, 13: 77,
         14: 145, 15: 90, 16: 60, 17: 97, 18: 40, 19: 90, 20: 45, 21: 55, 22: 60, 23: 58, 24: 84}
# FPL's current status/news in our data; everyone else is available with no news.
NEWS = {
    9: ("d", 75, "Knock - 75% chance of playing", "2026-09-20T18:00:00Z"),
    15: ("i", 0, "Groin injury - Expected back 18 Oct", "2026-09-14T10:00:00Z"),
    17: ("d", 50, "Hamstring - 50% chance of playing", "2026-09-12T09:00:00Z"),
}
# Availability the forecast assumed (Saka's knock came after it).
FORECAST_AVAILABILITY = {15: "injured", 17: "doubtful_50"}


def code(element):
    return 100000 + element


POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def status():
    """Latest price/status/news per player, as data.load_player_status()
    returns it from derived.db."""
    out = {}
    for e, *_ in PLAYERS:
        flag, chance, news, added = NEWS.get(e, ("a", None, "", None))
        out[code(e)] = {"now_cost": PRICE[e], "status": flag, "chance_of_playing_next_round": chance,
                        "news": news, "news_added": added, "fetched_at": DATA_AS_OF}
    return out


def universe():
    """The player list as data.load_player_universe() returns it from derived.db."""
    out = {}
    for e, w, f, s, t, et in PLAYERS:
        full, known = "{0} {1}".format(f, s), KNOWN_NAMES.get(e, "")
        out[code(e)] = {"code": code(e), "element": e, "web_name": w, "full_name": full,
                        "display_name": known or full, "known_name": known, "second_name": s,
                        "team": TEAMS[t - 1], "position": POSITIONS[et]}
    return out


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("{0} error".format(status), response=response)


def fetch_team_summary(team_id):
    if team_id != VALID_TEAM_ID:
        raise http_error(404)
    return {"id": team_id, "name": "Example XI", "player_first_name": "Alex", "player_last_name": "Example"}


def fetch_team_picks(team_id, event):
    if team_id != VALID_TEAM_ID or event > LAST_COMPLETED_GW:
        raise http_error(404)
    return {"picks": [{"element": e, "position": i + 1,
                       "multiplier": 0 if i >= 11 else (2 if e == CAPTAIN else 1),
                       "is_captain": e == CAPTAIN, "is_vice_captain": e == VICE_CAPTAIN}
                      for i, e in enumerate(SQUAD_ELEMENTS)],
            "entry_history": {"event": event, "bank": BANK, "value": 1000}}


def predictions(gameweek=LAST_COMPLETED_GW + 1, source=pstart.SOURCE_REGISTERED):
    """A pstart.PStartPredictions for every fake player, with CHANCE as
    p_start and a two-feature contribution breakdown per player."""
    teams = {i + 1: name for i, name in enumerate(TEAMS)}
    rows, contributions, explained = [], [], []
    for e, web, _, _, team, _ in PLAYERS:
        c = code(e)
        logit = math.log(CHANCE[e] / (1 - CHANCE[e]))
        positive, negative = logit / 2 + 0.5, logit / 2 - 0.5
        rows.append({"code": c, "web_name": web, "team": teams[team], "season": spec.PROSPECTIVE_SEASON,
                     "gameweek": gameweek, "p_start": CHANCE[e],
                     "availability_status": FORECAST_AVAILABILITY.get(e, "available"),
                     "last_gw_role": "started_60_plus",
                     "current_season_start_rate": e / 23, "previous_season_start_rate": None, "cold_start": False,
                     "logit": logit, "team_code": team})
        for group, label, what, effect in (("club_playing_time", "Playing time at his club", "started some games",
                                            -0.2 * (25 - e)), ("availability", "Availability", "available", 0.0)):
            p_if = 1 / (1 + pow(2.718281828459045, -(logit - effect)))
            explained.append({"code": c, "group": group, "label": label, "facts": what, "effect": effect,
                              "p_start_if_nailed_on": p_if, "gap": p_if - rows[-1]["p_start"]})
        for feature, value in (("current_season_start_rate", positive), ("minutes_prior_3_gws", negative)):
            contributions.append({"code": c, "feature": feature, "raw_feature": feature, "raw_value": e,
                                  "transformed_value": 1.0, "coefficient": value, "contribution": value,
                                  "description": spec.DESCRIPTIONS[feature]})
    metadata = {"model_id": spec.MODEL_ID, "source": source, "season": spec.PROSPECTIVE_SEASON,
                "gameweek": gameweek, "predicted_at": "20260920T100000Z", "prediction_cutoff": "2026-10-10T08:00:00Z",
                "deadline": "2026-10-10T10:00:00Z", "generated_after_deadline": False, "intercept": -1.0,
                "reference_p_start": 0.96, "reference_description": "a nailed-on starter"}
    explanation = pd.DataFrame(explained).sort_values(["code", "effect"], kind="stable").reset_index(drop=True)
    return pstart.PStartPredictions(pd.DataFrame(rows)[pstart.PLAYER_COLUMNS], pd.DataFrame(contributions), metadata,
                                    explanation)
