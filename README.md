# fpl-starts

An interpretable, leakage-safe P(start) modelling project for Fantasy
Premier League, with historical evaluation, calibration, prospective
scoring and explanation-ready predictions.

Fantasy Premier League has over 11 million players. Each week every one of
them decides which transfers to make and which 11 of their 15 players to
start, and almost every one of those decisions turns on whether a player
will actually start. FPL's own `chance_of_playing_next_round` (25/50/75/100)
only says whether a player *can* be picked; it says nothing about rotation,
form or a manager's preferences. This project estimates the probability
that each player starts his club's next match, and stores the reasoning
behind every number.

> **Modelling case study:** see
> [`notebooks/logistic_p_start_model.ipynb`](notebooks/logistic_p_start_model.ipynb)
> for the feature-selection, temporal-validation, calibration and
> player-level explainability walkthrough.

## The model

`src/fpl_starts/ml/` holds `logistic_availability`: a six-predictor logistic
regression (FPL availability status, last gameweek's role, minutes in the
three gameweeks before that, this season's and last season's start rate,
new-signing flag) built so every coefficient can be read on its own and
every prediction is stored with its full log-odds breakdown.

- Trained on 2023-24 to 2025-26 only, then frozen; 2026-27 is scored
  prospectively and never used to refit, retune or recalibrate it.
- On the development seasons it cuts Brier by 24.6% overall and 18.9% in
  the Rotation stratum against a naive P(start | started last gameweek).
- Every prediction snapshot stores, per player, each feature's raw value,
  transformed value, coefficient and contribution, plus the intercept,
  logit and probability -- ready for a conversational layer to explain.
- Needs local historical inputs under `data/` that are intentionally not
  distributed with this repository.

See `docs/logistic_p_start_model.md` for the operational reference
(commands, outputs, the rules around the frozen model, how to read a stored
prediction) and `docs/logistic_p_start_modelling_log.md` for the decisions
behind it.

## What's here

- **`api.py`** -- thin FPL API client.
- **`archiver.py`** -- write-once, gzip, manifest raw archive of
  `bootstrap-static`/`fixtures`/`event/{gw}/live`. This is the one component
  with an unrecoverable-if-missed deadline: availability fields
  (`status`, `chance_of_playing_next_round`, `news`) are live state with no
  history endpoint and no community-archive equivalent.
- **`derived.py`** -- SQLite layer (`teams`, `players`,
  `player_gameweek_stats`, `player_availability_snapshots`, `fixtures`,
  `predictions`), rebuilt from scratch by replaying the raw archive.
- **`ml/`** -- the logistic P(start) model: leakage-safe panel,
  preprocessing, walk-forward evaluation, train-once, and prediction
  snapshots with per-feature contributions.
- **`research/`** -- the aggregate analyses behind the case-study notebook.
- **`history.py`** -- start history across seasons (community archive +
  this project's own archive) and the write-once prediction-snapshot writer.
- **`scoring.py`** -- the calibration harness: Brier score and accuracy,
  stratified by how often a player actually starts (Core/Rotation/
  Marginal/Deep), against three baselines (persistence, season-rate,
  constant 0.9). `compare_models` scores several `model_version`s side by
  side against the same outcomes.
- **`quarantine.py`** -- moves any prediction snapshot whose `predicted_at`
  postdates its round's deadline out of the season directory, so a same-day
  rerun can never be mistaken for the real pre-deadline prediction.

## What's deliberately not here

News evidence (scraped team news, the adaptive search agent, and the news
adjustment of a base P(start)) and the full decision system built on top
of it live in a separate, private repository that consumes this package.
This repository is the statistical model and its evaluation.

## Running it

```
uv sync
```

### Before the deadline

```
uv run fpl-starts-archive                               # archive today's FPL data
uv run fpl-starts-derive                                # rebuild the derived SQLite layer
uv run fpl-starts-logistic-predict --target-round N     # frozen-model forecast for gameweek N
uv run fpl-starts-derive                                # rebuild again, to pick up predictions
```

### After the gameweek

```
uv run fpl-starts-archive                       # pick up the finished gameweek's results
uv run fpl-starts-derive                        # rebuild to pick up actual outcomes
uv run fpl-starts-score --target-round N        # score gameweek N, by stratum
```

### Modelling

```
uv run fpl-starts-logistic-evaluate                # historical walk-forward
uv run fpl-starts-logistic-train                   # fit and freeze (once)
uv run fpl-starts-logistic-case-study              # aggregates for the notebook
uv run --group notebook jupyter nbconvert --to notebook --execute --inplace \
    notebooks/logistic_p_start_model.ipynb
```

### Maintenance

```
uv run python -m fpl_starts.quarantine --season 2026-27 --dry-run
```

Check the dry run first: the 2026-27 GW1-5 logistic snapshots were
generated after their deadlines on purpose (retrospective replays, marked
`generated_after_deadline`), and a real run would move them out too.

Run from the repo root, these write `raw/` (the archive), `db/derived.db`
and `predictions/`. Each command takes `--help` for its full options; all
default to those relative paths, and an explicit flag always wins.

## Tests

```
uv run pytest                    # the root package, from the repo root (tests/)
cd dashboard && uv run pytest    # the dashboard's Streamlit AppTests (dashboard/tests/)
```

The dashboard tests need Streamlit, so they run separately in the
dashboard's own environment; the root `uv run pytest` collects `tests/`
only. Both use synthetic inputs -- no network, API key or local data.

## Dashboard

`dashboard/` -- a Streamlit dashboard + LangGraph agent for
`logistic_availability_v1`, read-only against this repo's registered
predictions, frozen model, `derived.db` and the public FPL API: per-player
P(start) with its explanation (the latest registered forecast, via
`fpl_starts.pstart`) for your current
squad -- your FPL team ID, validated, then the official squad from the last
completed gameweek plus the transfers you describe -- gameweek performance
against the baselines, and a chat interface that explains either. Its own nested Python project (separate `pyproject.toml`/venv --
see `dashboard/README.md`).
