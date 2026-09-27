# P(starts) dashboard

Streamlit dashboard + LangGraph agent for the frozen logistic P(start)
model, `logistic_availability_v1`, read-only against this repo's
`predictions/`, `models/`, `db/derived.db` and the public FPL API. Lives inside `fpl-starts` (the public
repo) as its own nested `pyproject.toml`/venv, separate from the parent
package's own -- not because of a Python-version conflict (both `fpl` and
`fpl-starts` moved to 3.12+ on 2026-09-15, specifically so this dashboard
didn't need a separate floor), but because a Streamlit/LangGraph app has
no business sharing a dependency set with the modelling pipeline it reads.
`fpl_starts` is consumed as a normal editable dependency (`path = ".."`).

## Setup

```
uv sync
uv run streamlit run app.py
```

Live predictions (see below) must be started from the repo root instead,
because the raw archive's manifest records paths relative to it:

```
cd ..
uv run --project dashboard streamlit run dashboard/app.py
```

The chat tab needs `ANTHROPIC_API_KEY` (and, only if your key isn't
scoped to a single workspace, `ANTHROPIC_WORKSPACE_ID` -- confirmed live:
an unscoped key 400s on every request without it). `agent.py` loads both
via `python-dotenv` automatically from a `.env` file in this directory,
or in the `fpl-starts` root (first one that sets a variable wins) -- put
one there yourself, it isn't provided. Deliberately doesn't look outside
this repo (a sibling private repo's `.env`, say) -- none of this is
required if the variables are already in your shell environment (`ant
auth login` also works, with no env var at all). `.env` files here are
gitignored.

`data.py` resolves `models/`, `predictions/`, `data/`, `raw/` and
`db/derived.db` from the repo root, whatever the working directory;
`FPL_DASHBOARD_FPL_STARTS_DB` overrides the database path. Nothing outside
this repo is read.

## Where P(start) comes from

Everything goes through `fpl_starts.pstart` -- the dashboard never builds
features or applies coefficients itself:

- **Registered snapshot** (default): the latest `logistic_availability_v1`
  snapshot in `predictions/`, written before the deadline by
  `fpl-starts-logistic-predict`. Only read, never written or replaced.
- **Live**: the frozen model (`models/`, loaded once) applied to the current
  pre-cutoff inputs (`db/derived.db`, `raw/`, `data/`) -- the same feature
  construction as `fpl-starts-logistic-predict`, but nothing is fitted or
  saved.

Each prediction carries its explanation: every feature's raw value,
coefficient and log-odds contribution. A missing model, missing inputs or a
gameweek with no prediction is an error on screen; there is no fallback to
any other model.

## What's here

- **`data.py`** -- all reads: P(start) via `fpl_starts.pstart`, scoring via
  `fpl_starts.scoring` (never a write), and the public, unauthenticated FPL
  manager-team API (`entry/{team_id}/event/{gw}/picks/`).
- **`agent.py`** -- the LangGraph agent (`claude-opus-5`). Two tools,
  `get_gameweek_report` and `get_team_squad_predictions`, both thin wrappers
  over `data.py` -- the model never estimates a probability or a score
  itself, same discipline as `fpl-starts`'s own agent challenger. `uv run
  python agent.py "your question"` for a quick CLI check outside Streamlit.
- **`app.py`** -- four tabs: per-player predictions (P(start), availability,
  last-GW role, start rates, top positive/negative factors, and a per-player
  breakdown), gameweek performance against the baselines (stratified), one
  manager's squad against the registered predictions (captain/vice-captain
  highlighted), and a chat interface wired to the agent.

## What this doesn't do (yet)

Read-only and explanatory only -- can't trigger the archive/derive/predict
pipeline, can't register a prediction, can't write anything back to
`derived.db` or `predictions/`. That was a deliberate scope decision (see the conversation
that produced this), not a missing feature; revisit if the read-only
agent proves useful and an operator-console tier is actually wanted.

## Tests

The P(start) service and `data.py` are covered by the root suite
(`tests/test_pstart.py`, run with `uv run pytest` from the repo root) on
synthetic inputs -- no Streamlit needed.
