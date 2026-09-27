# Who's likely to start? -- the chat decision layer

The decision layer of `fpl-starts`: a chat interface for FPL managers,
grounded in the frozen statistical model `logistic_availability_v1`. A
manager enters their FPL team ID, tells the app about any transfers since
the last gameweek, sees each player's chance of starting with the reason
behind it, and asks a LangGraph agent about their squad -- who's at risk,
who could replace whom within budget, who's injured. The agent answers only
through tools over the model and data: it phrases the answer, the model
supplies every number.

It lives inside `fpl-starts` as its own nested `pyproject.toml`/venv:
a Streamlit/LangGraph app has no business sharing a dependency set with the
modelling pipeline it reads. `fpl_starts` is consumed as a normal editable
dependency (`path = ".."`).

## Setup

```
uv sync
uv run streamlit run app.py
```

The chat needs `ANTHROPIC_API_KEY` (and, only if your key isn't scoped to a
single workspace, `ANTHROPIC_WORKSPACE_ID` -- an unscoped key is rejected
without it). `agent.py` loads both via `python-dotenv` from a `.env` file in
this directory or the `fpl-starts` root (first one that sets a variable
wins); it never looks outside this repo. None of this is needed if the
variables are already in your environment. `.env` files are gitignored.

`data.py` resolves `models/`, `predictions/`, `data/`, `raw/` and
`db/derived.db` from the repo root, whatever the working directory;
`FPL_DASHBOARD_FPL_STARTS_DB` overrides the database path. Nothing outside
this repo is read.

After changing anything other than `app.py`, restart the app -- Streamlit
re-runs `app.py` on every interaction but keeps imported modules loaded.

## Your squad first

The app opens by asking for your FPL team ID, checks it exists (FPL's
`entry/{id}/`), and loads your official squad as it stood at the end of the
last completed gameweek (`entry/{id}/event/{gw}/picks/`, which also gives
your bank). Those two calls, once per session, are the only per-user FPL
API requests: the player list (names, clubs, positions), prices, news and
the last completed gameweek come from `db/derived.db`, built from the
archive.

It then asks what you've changed since -- "No changes", "João Pedro out for
Calvert-Lewin", "sold A and B and bought C and D" -- and resolves each named
player to a stable FPL player code (accents and punctuation don't matter; an
unknown or ambiguous name, an outgoing player not in the squad or an
incoming one already in it is refused with nothing changed). Everything
after that -- the predictions, the player drill-down and the chat -- uses
that current squad: official squad + your transfers. The official squad
itself is never changed; "Change team" clears everything for the team.

## Where the chance of starting comes from

Everything goes through `fpl_starts.pstart` -- the dashboard never builds
features or applies coefficients itself. The app shows the latest registered
`logistic_availability_v1` forecast for the upcoming gameweek: the snapshot
in `predictions/` written by `fpl-starts-logistic-predict` or
`fpl-starts-refresh`. There are no source or gameweek options.

Each prediction is explained against a regular starter (available, played
60+ minutes last gameweek and all of the 3 before, started every game this
season and last -- 96% in the frozen model), by `fpl_starts.explanation`,
in three groups of inputs that belong together: **availability**, **playing
time at his club** (last gameweek, the 3 before, start rate this season,
first game at his club) and **last season**. Each group shows the facts
behind it, how much it's holding him back, and his chance without that
issue -- always a real, consistent player, never one input changed on its
own. The groups add up exactly to the prediction. In gameweeks 1-4, when
"last gameweek" still reaches into last season, the two playing-time groups
are shown as one.

## How the chat works

The chat is a LangGraph ReAct agent (`create_react_agent`, `claude-opus-5`).
Its graph is a loop: the **agent** node (Claude) either answers or asks for
a tool; the **tools** node runs it and hands the result back; repeat until
Claude answers. Claude never calculates a number itself -- every number in
an answer comes from a tool, and every tool answer says when our FPL data
is from.

One question, end to end:

```mermaid
sequenceDiagram
    actor M as Manager
    participant App as app.py
    participant G as LangGraph agent
    participant C as Claude (agent node)
    participant T as tools node
    participant Tools as tools.py
    M->>App: "Who's at risk in my squad?"
    App->>App: tools_context() - snapshot of the session and cached data
    App->>G: invoke(question)
    G->>C: question + tool descriptions
    C-->>G: call squad_risks()
    G->>T: run squad_risks (in a worker thread)
    T->>Tools: squad_risks(snapshot)
    Tools-->>T: "Starting XI risks ... FPL data as of ..."
    T-->>G: tool result
    G->>C: tool result
    C-->>G: final answer, using only the tool's numbers
    G-->>App: answer
    App->>App: keep_tool_writes() - copy a stated bank or reloaded forecast back
    App-->>M: answer (the page redraws if a refresh changed the data)
```

LangGraph runs tool calls in worker threads, where Streamlit's session state
and caches aren't available. So before each question `app.py` builds a
snapshot of the session and the cached data on its own thread, the tools
only ever touch that snapshot, and anything they change is copied back
afterwards.

The tools (`tools.py`, wrapped for LangGraph in `agent.py`):

| Tool | Answers | Built on |
|---|---|---|
| `get_my_current_squad_predictions` | "How's my team looking?" | forecast + explanation for the current squad |
| `explain_player` | "Why is Palmer only 80%?" -- any player | forecast, explanation, price and FPL news |
| `squad_risks` | "Who's at risk?" -- starters under 75%, and bench swaps that keep a legal formation | forecast + FPL's formation rules |
| `find_replacements` | "Who could replace Greaves for £5m?" | forecast, prices, bank (an estimate unless the user gives theirs), 3-per-club limit |
| `player_news` | "Is Saka fit?" -- and whether his status changed since the forecast | FPL news in derived.db vs the forecast's inputs |
| `refresh_fpl_data` | "Is this up to date?" | `fpl_starts.refresh` (below) |
| `get_gameweek_report` | "How did the model do in GW5?" | `fpl_starts.scoring` |

The agent only speaks to chance of starting: it has no model of points,
fixtures' difficulty or value, so it declines "who should I captain?" and
says so. The chat doesn't remember earlier questions yet.

## Keeping the data fresh

"Check for latest FPL news" and the chat's refresh tool run
`fpl_starts.refresh` -- for everyone using the app, not just the person who
asked:

```mermaid
flowchart TD
    Q["'Check for latest FPL news', or the chat: 'is this up to date?'"] --> D{"Before the upcoming<br/>gameweek's deadline?"}
    D -- no --> X["The forecast is final:<br/>nothing is fetched"]
    D -- yes --> C{"Refreshed in the<br/>last 30 minutes?"}
    C -- yes --> R["Keep the current data"]
    C -- no --> F["Fetch FPL bootstrap-static<br/>(one API call)"]
    F --> A["Archive it write-once in raw/"]
    A --> B["Rebuild derived.db atomically"]
    B --> P["Re-run the frozen model<br/>for the upcoming gameweek"]
    P --> CH{"Any player's chance<br/>of starting changed?"}
    CH -- yes --> N["Register a new forecast in predictions/<br/>and rebuild again"]
    CH -- no --> K["Keep the existing forecast"]
```

One refresh runs at a time (a lock file beside the database). Forecasts use
availability captured up to the deadline. Because it writes to `raw/`, `db/`
and `predictions/`, the app needs a writable disk.

## What's here

- **`app.py`** -- the onboarding steps, then the current squad's page:
  predictions across the full width (chance of starting, availability,
  last-GW role, start rates, and a per-player breakdown of what's holding him
  back), then the chat below them, with its input pinned to the bottom of
  the window so it's always in view.
- **`squad.py`** -- the onboarding state machine (`NO_TEAM` ->
  `TEAM_ID_VALID` -> `OFFICIAL_SQUAD_LOADED` -> `TRANSFER_STATE_CONFIRMED` ->
  `CURRENT_SQUAD_READY`), transfer parsing and player-name resolution.
  Framework-agnostic: it works on any mapping, `st.session_state` or a dict.
- **`tools.py`** -- the chat's tools, as plain functions over a snapshot of
  the session and the data.
- **`agent.py`** -- the LangGraph agent and its system prompt; wraps the
  tools for the app. `uv run python agent.py "your question"` for a quick
  check outside Streamlit.
- **`data.py`** -- all data access: the forecast via `fpl_starts.pstart`,
  players, prices, news and the last completed gameweek from
  `db/derived.db`, scoring via `fpl_starts.scoring`, the refresh via
  `fpl_starts.refresh`, and the two FPL manager-team API calls.

## What this doesn't do (yet)

- No points, captaincy or transfer-value advice -- only who's likely to start.
- No exact transfer budget: FPL doesn't publish selling prices, so the bank
  after transfers is an estimate unless the user gives theirs.
- No chat memory between questions.

## Tests

The root suite (`uv run pytest` from the repo root) covers the P(start)
service, `data.py`, the squad logic, every chat tool and the refresh end to
end on synthetic inputs -- no Streamlit, network or API key
(`tests/test_pstart.py`, `test_dashboard_squad.py`, `test_dashboard_tools.py`,
`test_refresh.py`). The app itself is covered by Streamlit AppTests
(`dashboard/tests/`; `uv run pytest` from this directory), including real
tool calls run through LangGraph by a scripted chat model, so no API calls
are made.
