# fpl-starts

Fantasy Premier League has over 11 million players. Each of them owns a
15-player squad and makes two decisions every week: which transfers to make,
and which 11 of the 15 to start. Both decisions turn on the same question --
will this player actually take the pitch. Get it wrong and the player scores
nothing from that slot -- no goal, no clean sheet, no bonus -- and if he's
captained, the doubled points are gone too. A transfer spent on someone who
then sits on the bench is a transfer spent on nothing.

The only signal the game gives for this is `chance_of_playing_next_round`, a
flag on the player's card (25/50/75/100). It measures fitness, not
selection -- it can't tell a nailed-on starter from a fourth-choice option,
because that was never what it was built to answer.

This package automates a P(starts) estimate to fill that gap: not the flag
read at face value, but the observed frequency of starting, backtested
against real outcomes and reported honestly for the players it's actually
hard to call, not just the easy ones.

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
- **`starts_model.py`** -- the model. `predict_gameweek`: a lookup table
  crossing `prev` (started last gameweek?) and `roll4` (share of the last 4
  started), observed frequency per cell -- no regression, no fitting.
  `predict_gameweek_refined`: layers FPL's own `chance_of_playing_next_round`
  on top as a gate (hard-zero for injured/suspended, an observed-frequency
  flag table for doubtful players) -- deliberately never
  `P(available) * P(selected)`.
- **`scoring.py`** -- the calibration harness: Brier score and accuracy,
  stratified by how often a player actually starts (Core/Rotation/
  Marginal/Deep), against three baselines (persistence, season-rate,
  constant 0.9). `compare_models` scores several `model_version`s side by
  side against the same outcomes.

## Why a lookup table, not a fitted model

Two features, seven populated cells, observed frequency per cell. Measured
walk-forward on a full season (27 folds): beats a persistence baseline
("predict what they did last week") in 96% of individual gameweeks, and a
Beta-Binomial shrinkage toward per-player history was tested and added
nothing -- the optimal weight on player-specific history beyond `prev`/
`roll4` turned out to be approximately zero. See
`docs/build_spec_p_starts.md` (the design spec) and
`notebooks/fpl_starts_analysis.ipynb` (the analysis that produced it) for
the full case, including why a single pool-wide Brier score is actively
misleading (Deep/never-starting players are ~41% of rows and trivially
predictable, and dominate any pool average) and why Rotation-stratum Brier
is the number that actually matters.

## What's deliberately not here

This package is the base model only -- two baselines (`raw_lookup` and
`refined_availability`) plus the archiver/derived layer/calibration harness
that support them, kept deliberately minimal. An AI-agent-based challenger
to these baselines (explicit tool calls, including its own
evidence-gathering, evaluated through the same `scoring.py` harness) is the
planned next addition to this repo.

## Running the pipeline

```
uv sync
uv run fpl-starts-archive          # archive today's FPL data
uv run fpl-starts-derive           # rebuild the derived SQLite layer
uv run fpl-starts-predict          # predict the next unplayed gameweek
uv run fpl-starts-derive           # rebuild again, to pick up predictions
uv run fpl-starts-score --target-round N   # once gameweek N is played
```

Each command takes `--help` for its full options (`--base-dir`, `--db-path`,
`--season`, `--target-round`, etc.) -- all default to sensible relative
paths from wherever you run them.

## Tests

```
uv run pytest
```
