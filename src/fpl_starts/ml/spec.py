"""The frozen specification of the logistic P(start) model.

Everything a fitted model, a prediction or an explanation needs to agree on
lives here: which seasons may be used for what, the six predictors, their
encodings, the fixed category levels and reference categories, and the
human-readable descriptions the explanation layer surfaces.

Changing anything in this file changes the model. A changed model is a new
`MODEL_ID`, never an edit to an existing fitted one.
"""

MODEL_VERSION = "logistic_availability"  # stable public name (predictions.model_version)
MODEL_ID = "logistic_availability_v1"    # this exact specification

# --- Temporal boundary -------------------------------------------------------
#
# 2022-23 is the first season of the historical panel. Its rows have no
# previous-season history and, early in the season, every player looks like a
# new arrival at his club -- artefacts of where the panel starts, not football.
# Those rows distort coefficients and calibration, so 2022-23 only supplies
# history (previous-season rates, recent form) for 2023-24's features.
#
# 2023-24 to 2025-26 are model-development history: the feature specification
# was chosen by walk-forward comparison on 2024-25 and 2025-26, and the frozen
# model is fitted on all three. None of them is an untouched test set.
#
# 2026-27 is untouched prospective evaluation: the frozen model scores it
# without ever being refitted, retuned or recalibrated on it.
CONTEXT_ONLY_SEASONS = ("2022-23",)
TRAINING_SEASONS = ("2023-24", "2024-25", "2025-26")
HISTORICAL_SEASONS = CONTEXT_ONLY_SEASONS + TRAINING_SEASONS
PROSPECTIVE_SEASON = "2026-27"

# Prediction cutoff: two hours before each gameweek's deadline. Only
# information timestamped strictly before it may describe that gameweek.
CUTOFF_HOURS_BEFORE_DEADLINE = 2

# --- Predictors ----------------------------------------------------------------

CONTINUOUS = [
    "minutes_prior_3_gws",
    "current_season_start_rate",
    "previous_season_start_rate",
]
BINARY = [
    "no_previous_season",
    "first_game_at_club",
]
# name -> (reference level, other levels in fixed order). Levels are fixed
# here rather than learned from data so the transformed matrix is identical
# in every fold, every fit and every prediction.
CATEGORICAL = {
    "availability_status": ("available", [
        "doubtful_75", "doubtful_50", "doubtful_25",
        "injured", "suspended", "unavailable", "unknown",
    ]),
    "last_gw_role": ("did_not_play", [
        "sub_appearance", "started_under_60", "started_60_plus",
    ]),
}

RAW_FEATURES = ["availability_status", "last_gw_role"] + CONTINUOUS + BINARY


def transformed_feature_names():
    """Column order of the model matrix -- the order coefficients are stored
    in. Categorical columns are `<feature>__<level>`, reference dropped."""
    names = []
    for feature, (_, levels) in CATEGORICAL.items():
        names.extend("{0}__{1}".format(feature, level) for level in levels)
    names.extend(CONTINUOUS)
    names.extend(BINARY)
    return names


DESCRIPTIONS = {
    "availability_status": "FPL availability flag as of the prediction cutoff",
    "availability_status__doubtful_75": "flagged doubtful, 75% chance of playing (vs available)",
    "availability_status__doubtful_50": "flagged doubtful, 50% chance of playing (vs available)",
    "availability_status__doubtful_25": "flagged doubtful, 25% chance of playing (vs available)",
    "availability_status__injured": "flagged injured (vs available)",
    "availability_status__suspended": "flagged suspended (vs available)",
    "availability_status__unavailable": "flagged unavailable, e.g. left the club or on loan (vs available)",
    "availability_status__unknown": "no availability record before the cutoff (vs available)",
    "last_gw_role": "role in the player's previous gameweek at his current club",
    "last_gw_role__sub_appearance": "came off the bench last gameweek (vs did not play)",
    "last_gw_role__started_under_60": "started last gameweek but played under 60 minutes (vs did not play)",
    "last_gw_role__started_60_plus": "started last gameweek and played 60+ minutes (vs did not play)",
    "minutes_prior_3_gws": "minutes played in the three gameweeks before last, at his current club",
    "current_season_start_rate": "share of this season's gameweeks at his current club that he started",
    "previous_season_start_rate": "share of last season's gameweeks that he started",
    "no_previous_season": "no Premier League season last year (new to the league or promoted)",
    "first_game_at_club": "no previous gameweek at his current club (new signing)",
}

# How a missing continuous value is filled. Replaced by the training mean, it
# transforms to exactly 0 and so contributes nothing; the two semantic flags
# above carry the "no history" effect on their own.
MISSING_VALUE_RULE = "training mean (transforms to 0, contributes 0 to the logit)"

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
