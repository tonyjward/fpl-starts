"""Synthetic FPL API payloads and P(start) predictions for the dashboard
tests -- invented data shaped like bootstrap-static, entry/ and picks/,
never real API responses. Shared by the root suite's squad-logic tests
(tests/test_dashboard_squad.py) and the Streamlit AppTests here."""

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
    (4, "Saliba", "William", "Saliba", 1, 2),
    (5, "Virgil", "Virgil", "van Dijk", 4, 2),
    (6, "Gvardiol", "Joško", "Gvardiol", 5, 2),
    (7, "Muñoz", "Daniel", "Muñoz Mejía", 2, 2),
    (8, "B.Fernandes", "Bruno", "Borges Fernandes", 6, 3),
    (9, "Saka", "Bukayo", "Saka", 1, 3),
    (10, "Mbeumo", "Bryan", "Mbeumo", 6, 3),
    (11, "Gordon", "Anthony", "Gordon", 7, 3),
    (12, "Rice", "Declan", "Rice", 1, 3),
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
]
TEAMS = ["Arsenal", "Crystal Palace", "Everton", "Liverpool", "Man City", "Man Utd", "Newcastle", "Chelsea",
         "Leeds", "Ipswich", "Aston Villa", "West Ham", "Fulham"]
SQUAD_ELEMENTS = list(range(1, 16))


def code(element):
    return 100000 + element


POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def universe():
    """The player list as data.load_player_universe() returns it from derived.db."""
    return {code(e): {"code": code(e), "element": e, "web_name": w, "full_name": "{0} {1}".format(f, s),
                      "known_name": "", "second_name": s, "team": TEAMS[t - 1], "position": POSITIONS[et]}
            for e, w, f, s, t, et in PLAYERS}


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
    return {"picks": [{"element": e, "position": i + 1, "multiplier": 0 if i >= 11 else (2 if e == 14 else 1),
                       "is_captain": e == 14, "is_vice_captain": e == 9} for i, e in enumerate(SQUAD_ELEMENTS)]}


def predictions(gameweek=LAST_COMPLETED_GW + 1, source=pstart.SOURCE_REGISTERED):
    """A pstart.PStartPredictions for every fake player, with a distinct
    p_start and a two-feature contribution breakdown per player."""
    teams = {i + 1: name for i, name in enumerate(TEAMS)}
    rows, contributions, explained = [], [], []
    for e, web, _, _, team, _ in PLAYERS:
        c = code(e)
        positive, negative = 0.1 * e, -0.05 * (24 - e)
        logit = -1.0 + positive + negative
        rows.append({"code": c, "web_name": web, "team": teams[team], "season": spec.PROSPECTIVE_SEASON,
                     "gameweek": gameweek, "p_start": 1 / (1 + pow(2.718281828459045, -logit)),
                     "availability_status": "available", "last_gw_role": "started_60_plus",
                     "current_season_start_rate": e / 23, "previous_season_start_rate": None, "cold_start": False,
                     "logit": logit, "team_code": team})
        for group, label, what, effect in (("club_playing_time", "Playing time at his club", "started some games",
                                            -0.2 * (24 - e)), ("availability", "Availability", "available", 0.0)):
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
