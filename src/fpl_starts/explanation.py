"""Grouped explanations of the frozen logistic P(start) model, compared with a
nailed-on starter.

The model's stored contributions are reference-coded: a categorical level
such as last_gw_role = did_not_play has no coefficient and contributes 0
however much it matters, and single inputs can't be changed on their own
without describing a player who can't exist (a 100% start rate this season
but didn't play last gameweek). So a prediction is explained by comparing
the player with one fixed, realistic reference -- a nailed-on starter -- and
by changing inputs only in groups that belong together:

    effect_g = sum over g's model columns j of  coefficient_j * (x_j - ref_j)
    reference_logit = intercept + sum_j coefficient_j * ref_j

where ref_j is the nailed-on starter's value of model column j. Then
reference_logit + the group effects = the player's logit, exactly, and each
group's "if this matched a nailed-on starter" figure describes a player who
could exist.

Groups: availability; playing time at his club (last gameweek, minutes in
the three before, start rate this season, first game at his club); last
season (start rate, no Premier League season). In gameweeks 1-4 "last
gameweek" and "the three before" reach back into last season, whose games
also make up last season's start rate, so there the two playing-time groups
are merged into one.

Everything here comes from the frozen model (model.json) and the stored
contributions; the model itself is not touched. This lives outside
`fpl_starts.ml`, whose source files the frozen model fingerprints.
"""

import math

import pandas as pd

from .ml import spec

# The reference player: in every input, what a nailed-on starter looks like.
NAILED_ON = {
    "availability_status": "available",
    "last_gw_role": "started_60_plus",
    "minutes_prior_3_gws": 270.0,
    "current_season_start_rate": 1.0,
    "previous_season_start_rate": 1.0,
    "no_previous_season": 0,
    "first_game_at_club": 0,
}
NAILED_ON_DESCRIPTION = ("available, played 60+ minutes last gameweek and all of the 3 before, "
                         "and started every game this season and last")

AVAILABILITY = "availability"
CLUB_PLAYING_TIME = "club_playing_time"
LAST_SEASON = "last_season"
PLAYING_TIME = "playing_time"  # the two playing-time groups merged, gameweeks 1-4

GROUPS = {
    AVAILABILITY: ["availability_status"],
    CLUB_PLAYING_TIME: ["last_gw_role", "minutes_prior_3_gws", "current_season_start_rate", "first_game_at_club"],
    LAST_SEASON: ["previous_season_start_rate", "no_previous_season"],
}
assert sorted(f for fs in GROUPS.values() for f in fs) == sorted(spec.RAW_FEATURES)

GROUP_LABELS = {
    AVAILABILITY: "Availability",
    CLUB_PLAYING_TIME: "Playing time at his club",
    LAST_SEASON: "Last season",
    PLAYING_TIME: "Playing time",
}
# Up to this gameweek the recent-playing-time windows overlap last season.
MERGED_UP_TO_GAMEWEEK = 4

AVAILABILITY_TEXT = {
    "available": "available",
    "doubtful_75": "flagged doubtful (75% chance of playing)",
    "doubtful_50": "flagged doubtful (50% chance of playing)",
    "doubtful_25": "flagged doubtful (25% chance of playing)",
    "injured": "injured",
    "suspended": "suspended",
    "unavailable": "unavailable (e.g. left the club or on loan)",
    "unknown": "no availability information",
}
LAST_GW_TEXT = {
    "started_60_plus": "played 60+ minutes last gameweek",
    "started_under_60": "started last gameweek but played under 60 minutes",
    "sub_appearance": "came off the bench last gameweek",
    "did_not_play": "didn't play last gameweek",
}


def reference_values(model):
    """The nailed-on starter's value in every model column, and his logit."""
    ref = model.transform(pd.DataFrame([NAILED_ON])).iloc[0]
    return ref, float(model.intercept + (model.coefficients * ref[model.coefficients.index]).sum())


def group_of(raw_feature, gameweek):
    for group, features in GROUPS.items():
        if raw_feature in features:
            if group != AVAILABILITY and int(gameweek) <= MERGED_UP_TO_GAMEWEEK:
                return PLAYING_TIME
            return group
    raise KeyError(raw_feature)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _club_facts(raw):
    if raw.get("first_game_at_club") == 1:
        return ["first game at his club"]
    facts = [LAST_GW_TEXT.get(raw.get("last_gw_role"), str(raw.get("last_gw_role")))]
    minutes = raw.get("minutes_prior_3_gws")
    if not _missing(minutes):
        facts.append("{0:.0f} minutes in the 3 gameweeks before".format(minutes))
    rate = raw.get("current_season_start_rate")
    facts.append("no earlier games this season" if _missing(rate)
                 else "started {0:.0%} of his games this season".format(rate))
    return facts


def _last_season_facts(raw):
    if raw.get("no_previous_season") == 1:
        return ["not in the Premier League last season"]
    rate = raw.get("previous_season_start_rate")
    return ["started {0:.0%} of games last season".format(rate)] if not _missing(rate) else []


def facts(group, raw):
    """Plain-English facts behind one group, from the player's raw inputs."""
    if group == AVAILABILITY:
        return [AVAILABILITY_TEXT.get(raw.get("availability_status"), str(raw.get("availability_status")))]
    if group == CLUB_PLAYING_TIME:
        return _club_facts(raw)
    if group == LAST_SEASON:
        return _last_season_facts(raw)
    return _club_facts(raw) + _last_season_facts(raw)


# A group whose "if like a nailed-on starter" chance is within this many
# percentage points of the player's own chance is "in line" with one. Sizes
# are judged on this percentage-point gap -- what users see -- rather than
# on log-odds, where the same effect is a large change at 50% and a tiny one
# at 95%.
NEGLIGIBLE_GAP = 0.02


def impact(gap):
    """Plain-English size and direction of a group's `gap`: its chance if
    like a nailed-on starter minus the player's own chance."""
    if gap >= 0.25:
        return "Holding him back a lot"
    if gap >= 0.10:
        return "Holding him back"
    if gap >= NEGLIGIBLE_GAP:
        return "Holding him back a little"
    if gap > -NEGLIGIBLE_GAP:
        return "In line with a nailed-on starter"
    return "Helping him"


def summary(rows):
    """One line of text for a player's grouped explanation (rows from
    `grouped_explanation`): what is holding him back, and what he'd be
    without each."""
    holding = rows[rows["gap"] >= NEGLIGIBLE_GAP]
    if holding.empty:
        return "nothing is holding him back compared with a nailed-on starter"
    return "held back by " + "; ".join(
        "{0} ({1}; would be {2:.0%} if like a nailed-on starter)".format(r.label.lower(), r.facts, r.p_start_if_nailed_on)
        for r in holding.itertuples())


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def grouped_explanation(contributions, logits, gameweek, model):
    """One row per (player, group), from stored reference-coded
    `contributions` (pstart format) and each player's logit {code: logit}:
    the group's plain-English facts, its effect on the logit compared with a
    nailed-on starter, the player's chance were that group like a nailed-on
    starter's, and the gap between that and his own chance. Rows are ordered
    most-limiting group first."""
    ref, _ = reference_values(model)
    c = contributions[["code", "feature", "raw_feature", "raw_value", "coefficient", "contribution"]].copy()
    c["effect"] = c["contribution"] - c["coefficient"] * c["feature"].map(ref)
    c["group"] = [group_of(f, gameweek) for f in c["raw_feature"]]
    raw = {code: dict(zip(g["raw_feature"], g["raw_value"])) for code, g in c.groupby("code", sort=False)}
    rows = []
    for (code, group), g in c.groupby(["code", "group"], sort=False):
        effect = float(g["effect"].sum())
        p_if = sigmoid(logits[code] - effect)
        rows.append({"code": code, "group": group, "label": GROUP_LABELS[group],
                     "facts": "; ".join(facts(group, raw[code])), "effect": effect,
                     "p_start_if_nailed_on": p_if, "gap": p_if - sigmoid(logits[code])})
    out = pd.DataFrame(rows, columns=["code", "group", "label", "facts", "effect", "p_start_if_nailed_on", "gap"])
    return out.sort_values(["code", "effect"], kind="stable").reset_index(drop=True)
