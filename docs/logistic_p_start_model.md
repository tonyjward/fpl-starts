# Logistic P(start): reference

`logistic_availability` predicts the probability that a player starts his
club's next Premier League gameweek, with a six-predictor logistic
regression whose every prediction is stored with the arithmetic behind it.
This page is the operational reference: how to run it, what it writes, the
rules around the frozen model, and how to read a stored prediction.

- **Why it looks the way it does**, with figures and two worked
  predictions: [`notebooks/logistic_p_start_model.ipynb`](../notebooks/logistic_p_start_model.ipynb).
- **The decisions, in order, with the evidence for each** (the audit
  trail): [`docs/logistic_p_start_modelling_log.md`](logistic_p_start_modelling_log.md).
- **The frozen specification**: `src/fpl_starts/ml/spec.py`. Code:
  `src/fpl_starts/ml/`.

## Predictors

| Predictor | Encoding | Reference level |
|---|---|---|
| `availability_status` | categorical: `doubtful_75`, `doubtful_50`, `doubtful_25`, `injured`, `suspended`, `unavailable`, `unknown` | `available` |
| `last_gw_role` | categorical: `sub_appearance`, `started_under_60`, `started_60_plus` | `did_not_play` |
| `minutes_prior_3_gws` | continuous, standardised | training mean |
| `current_season_start_rate` | continuous, standardised | training mean |
| `previous_season_start_rate` + `no_previous_season` | continuous, standardised; flag 0/1 | training mean; 0 |
| `first_game_at_club` | flag 0/1 | 0 |

15 coefficients plus an intercept. A missing continuous value is replaced
by the training mean, so it contributes 0.

## Rules

- **Seasons.** 2022-23 supplies history only. 2023-24 to 2025-26 are
  training (and development) seasons. 2026-27 is prospective: the frozen
  model is never refitted, retuned or recalibrated on it.
- **Cutoff.** Features for gameweek *N* use only information from before
  *N*'s prediction cutoff (deadline − 2h). Later availability records are
  rejected.
- **Frozen means frozen.** `fpl-starts-logistic-train` refuses to overwrite
  an existing model. Any change to `spec.py` is a new `MODEL_ID` with its
  own prospective record, never an edit to `logistic_availability_v1`.
- **Provenance.** `model.json` records a hash of every source file in
  `src/fpl_starts/ml/`. Research and analysis code lives elsewhere
  (`src/fpl_starts/research/`) so it cannot change that hash.
- **Two kinds of 2026-27 evidence.** Snapshots generated after the deadline
  (`generated_after_deadline: true`) are retrospective replays; the rest
  are real forecasts. They are scored and reported separately.

## Commands

```
uv run fpl-starts-logistic-evaluate                # historical walk-forward, development seasons
uv run fpl-starts-logistic-train                   # fit and freeze the model (once)
uv run fpl-starts-logistic-predict --target-round 7
uv run fpl-starts-logistic-case-study              # aggregates behind the notebook
uv run --group notebook jupyter nbconvert --to notebook --execute --inplace \
    notebooks/logistic_p_start_model.ipynb
```

## Outputs

| File | Written by | Contents |
|---|---|---|
| `models/logistic_availability/logistic_availability_v1/model.json` | train | coefficients, intercept, preprocessing statistics, reference levels, feature order, a plain-language description of every column, training seasons and row counts, the `C` search, input, training-matrix and source fingerprints, library versions |
| `models/logistic_availability/logistic_availability_v1/historical_evaluation.json` | train, evaluate | per-fold coefficients and `C`, Brier by season and stratum, calibration bins |
| `models/logistic_availability/case_study.json` | case-study | aggregates for the notebook: candidate correlations, VIFs, comparison fits, performance by player type, the benchmark's lookup table |
| `predictions/<season>/gwNN_logistic_availability_<timestamp>.json` | predict | one write-once snapshot per run |

A prediction snapshot uses the same format as the other models, so
`fpl-starts-derive` loads it and `fpl-starts-score` scores it. It adds, per
player:

- `explanations[].features[]`: `feature`, `raw_feature`, `raw_value`,
  `transformed_value`, `coefficient`, `contribution`, for every model column;
- `intercept`, `logit`, `p_start`;
- the model id and file hash, the prediction cutoff, and
  `generated_after_deadline`.

`models/` and `predictions/` are gitignored.

## Reading a stored prediction

    logit   = intercept + sum_j contribution_j,   contribution_j = coefficient_j * x_j
    p_start = 1 / (1 + exp(-logit))

- **Log-odds, not percentage points.** A contribution of −1.7 multiplies
  the odds of starting by about e^−1.7 ≈ 0.18; how many percentage points
  that removes depends on where the player started from.
- **Categories are relative to their reference.** `doubtful_50` is the
  effect of being doubtful at 50% compared with being available; an
  available player's status contributes exactly 0.
- **Continuous features are per standard deviation, relative to the
  training average.** The stored training mean and SD convert back to real
  units.
- **The intercept is a reference player**: available, did not play last
  gameweek, average on every continuous feature, with a previous season
  and not new at his club.
- **Read `first_game_at_club` together with the defaults it offsets**
  (no last-gameweek role, average form), not on its own.
- Comparing two snapshots for the same player answers "what changed since
  last week" column by column.

## Data

Training and scoring need historical inputs under `data/` (see
`fpl_starts.ml.data` for the expected layout): per-season player match
histories, pre-deadline availability, and historical deadlines. **They are
intentionally not distributed with this repository.** Without them every
command stops with a clear error rather than training or predicting without
availability. `data/` is gitignored.
