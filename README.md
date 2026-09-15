# fpl-starts

Fantasy Premier League has over 11 million players. Each of them owns a
15-player squad and makes two decisions every week: 
* which transfers to make,
* which 11 of the 15 to start. 

Having a robust estimate of the probability of starting the next game
is valuable information for FPL players, as it allows better decisions.
You don't want to captain/buy/transfer in players that are not going to be
involved in the game.

The FPL app does provide a `chance_of_playing_next_round` which takes
values (25/50/75/100). However this only measures whether a player
is available to be picked (injured players are less likely). It does
not tell us if the manager will drop the player due to poor performance,
or to rotate the squad.

The aim of this project is to produce P(start) for all Premier League
players. This repo tests 3 model variants:
1) Only a player's recent history (`raw_lookup`).
2) Layer on FPL's own injury status flag on top of 1 (`refined_availability`).
3) Layer on a team news adjustment to 2, gathered by an AI agent
   (`refined_availability_agent_news`).

See `docs/p_starts_model_architecture.md` for how the three fit together,
including a worked example for each and how the agent's claim priors work
and change over time.

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
- **`agent/`** -- the AI-agent challenger. `predict.py`: a manual ReAct loop
  (predates this project's 2026-09-15 Python floor bump, which lifted the
  SDK-version ceiling that originally forced this -- kept as-is since
  migrating to native tool-calling wasn't itself the goal of that change,
  see the module docstring) that gives an LLM its own web-search and
  page-reading tools, one club roster at a time, and asks it to classify
  each player into a fixed taxonomy (`categories.py`) -- never a
  probability directly. `domain_stats.py`: per-source-domain accuracy
  monitoring for the evidence the agent gathers (monitoring only for now --
  doesn't yet feed back into the blend).
- **`quarantine.py`** -- moves any archived prediction snapshot whose
  `predicted_at` postdates its round's deadline out of the season
  directory, so a same-day dev/test rerun can never silently get treated as
  the real pre-deadline prediction (`derived.py` otherwise trusts whichever
  snapshot is *latest*).
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

The news-scraping/LLM-extraction evidence layer that predates the agent
(scraped articles classified into the same taxonomy, plus source-tier
weighting and claim-level scoring) lives in a separate, private repo that
consumes this package as an editable dependency -- kept out of this repo
to keep the base model + agent challenger minimal and independently
useful. `docs/p_starts_model_architecture.md`'s "Where things live" table
covers only this repo's own three arms.

Also not yet here: the agent's per-domain accuracy (`domain_stats.py`)
doesn't feed back into `predict.py`'s blend -- a source found unreliable
is visible in a report, but doesn't yet lose influence. That's a real,
identified next step, not an oversight.

## Running the pipeline

```
uv sync
uv run fpl-starts-archive                       # archive today's FPL data
uv run fpl-starts-derive                        # rebuild the derived SQLite layer
uv run fpl-starts-predict                       # predict the next unplayed gameweek
uv run fpl-starts-agent-predict --team "Arsenal" # agent challenger, one club roster at a time
uv run fpl-starts-derive                        # rebuild again, to pick up predictions
uv run fpl-starts-score --target-round N        # once gameweek N is played
uv run fpl-starts-agent-report                  # per-domain accuracy for the agent's evidence
uv run python -m fpl_starts.quarantine --season 2026-27  # move out any post-deadline snapshot
```

Each command takes `--help` for its full options (`--base-dir`, `--db-path`,
`--season`, `--target-round`, etc.) -- all default to sensible relative
paths from wherever you run them.

## Tests

```
uv run pytest
```

## Dashboard

`dashboard/` -- a Streamlit dashboard + LangGraph agent, read-only against
this repo's and the private news repo's `derived.db` and the public FPL
API: gameweek performance across every arm, one manager's squad (by FPL
team ID) against P(starts) predictions, and a chat interface that explains
either. Its own nested Python project (separate `pyproject.toml`/venv,
not a Python-version workaround -- see `dashboard/README.md`).
