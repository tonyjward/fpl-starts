# P(starts) dashboard

Streamlit dashboard + LangGraph agent, read-only against both repos'
`derived.db` and the public FPL API. Lives inside `fpl-starts` (the public
repo) as its own nested `pyproject.toml`/venv, separate from the parent
package's own -- not because of a Python-version conflict (both `fpl` and
`fpl-starts` moved to 3.12+ on 2026-09-15, specifically so this dashboard
didn't need a separate floor), but because a Streamlit/LangGraph app has
no business sharing a dependency set with the modelling pipeline it reads.
`fpl_starts` is consumed as a normal editable dependency (`path = ".."`).

## Setup

```
uv sync
export ANTHROPIC_API_KEY=...   # or `ant auth login` -- needed for the chat tab only
uv run streamlit run app.py
```

Run from this directory -- `data.py`'s default DB paths (`../derived.db`
for this repo's own, `../../fpl/derived.db` for the private news repo's)
are relative to it. Override with `FPL_DASHBOARD_FPL_STARTS_DB` /
`FPL_DASHBOARD_NEWS_DB` if you run it from somewhere else, or don't have
the private repo checked out at all -- the fpl-starts side works
standalone.

## What's here

- **`data.py`** -- all reads: both `derived.db` files (via `fpl_starts.scoring`/
  plain SQL, never a write), and the public, unauthenticated FPL manager-team
  API (`entry/{team_id}/event/{gw}/picks/`).
- **`agent.py`** -- the LangGraph agent (`claude-opus-5`). Two tools,
  `get_gameweek_report` and `get_team_squad_predictions`, both thin wrappers
  over `data.py` -- the model never estimates a probability or a score
  itself, same discipline as `fpl-starts`'s own agent challenger. `uv run
  python agent.py "your question"` for a quick CLI check outside Streamlit.
- **`app.py`** -- three tabs: gameweek performance (every arm, stratified),
  one manager's squad against those predictions (captain/vice-captain
  highlighted), and a chat interface wired to the agent.

## What this doesn't do (yet)

Read-only and explanatory only -- can't trigger the archive/derive/predict
pipeline, can't adjust a prior, can't write anything back to either
`derived.db`. That was a deliberate scope decision (see the conversation
that produced this), not a missing feature; revisit if the read-only
agent proves useful and an operator-console tier is actually wanted.

## Tests

None yet beyond the manual smoke checks this was built with (`data.py`'s
functions run against real archived predictions; `agent.py`'s graph
constructs cleanly; the Streamlit app starts and serves). No live LLM call
has been verified end-to-end in this environment -- no `ANTHROPIC_API_KEY`
was available when this was built.
