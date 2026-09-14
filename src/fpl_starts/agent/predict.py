"""The agent challenger: classifies each player on one club's roster into a
fixed news taxonomy, using its own tool calls to gather evidence, then hands
the classification (never a probability -- see categories.py) to a
deterministic lookup for the final p_start.

**Explicit tool calls in code, not a framework.** The Anthropic SDK version
this project's Python 3.7 target can install (anthropic==0.26.0, pinned by
the same tokenizers<0.20 constraint the private repo already needed) predates
native `tools=`/tool_use support in this SDK. Rather than pull in a newer
SDK a consuming environment can't run, the loop below is a plain ReAct-style
loop this code owns end to end: the model is asked to respond with one JSON
action per turn (search_web / fetch_page_text / final_answer), this module
parses that JSON, executes the requested tool itself, and feeds the result
back as the next turn -- no framework mediates any of this, arguably more
explicit than the SDK's own tool-calling sugar would have been.

**Classify, don't estimate.** The private repo tested an arm where the LLM
emitted a probability directly against an arm where it classified into a
fixed taxonomy and a separate lookup priced the category from observed
outcomes -- twice (a retrospective pilot and a live gameweek), the
category-based arm won both times, because an LLM-invented float is an
ungrounded, unauditable number. This agent's model output is *only ever* a
category plus a verbatim quote; categories.py, not the model, decides what
that category is worth.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import json
import os

import pandas as pd

from fpl_starts import starts_model
from fpl_starts.agent import categories
from fpl_starts.agent import tools as agent_tools
from fpl_starts.agent.tools import ToolBudget, ToolBudgetExceeded

DEFAULT_MODEL = "claude-sonnet-4-5"

TAXONOMY_DESCRIPTIONS = (
    "confirmed_starting -- a manager, press conference, or predicted lineup "
    "names this player in the starting XI for the upcoming fixture.\n"
    "confirmed_out -- the player is ruled out (injury, suspension, "
    "explicitly dropped) for the upcoming fixture.\n"
    "rotation_risk -- credible reporting that the player may be rested or "
    "rotated, short of an outright confirmed absence.\n"
    "returning_from_injury -- the player is recovering and may or may not "
    "be risked, per reporting."
)

SYSTEM_PROMPT = """You are classifying Fantasy Premier League players ahead \
of their club's next match, using web search and page-reading tools.

For each player on the roster you are given, decide whether there is \
credible, cited evidence placing them in one of these categories:

{taxonomy}

A player with no supporting evidence gets no classification at all -- do \
not guess. Never estimate a probability yourself; only ever return a \
category and the exact quote that supports it.

Respond with exactly one JSON object per turn, and nothing else -- no \
markdown fences, no prose outside the JSON. Valid actions:

{{"action": "search_web", "query": "..."}}
{{"action": "fetch_page_text", "url": "..."}}
{{"action": "final_answer", "classifications": [
  {{"code": <player code, integer>, "category": "<one of the categories above>", \
"quote": "<verbatim quote from a fetched page>", "source_url": "<the url that quote came from>"}}
]}}

Every quote must be copied verbatim from a page you fetched with \
fetch_page_text -- not paraphrased, not from a search snippet alone. \
Omit any player you found no evidence for; they are handled separately. \
Call final_answer once you've covered the roster or have used your \
available tool calls.""".format(taxonomy=TAXONOMY_DESCRIPTIONS)


def _strip_json_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_response_text(message):
    """Plain-text content of one Anthropic Message response -- this loop
    never uses native tool_use content blocks (see module docstring), so
    the response is always a single text block.
    """
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def load_club_roster(conn, season, prior_season, target_round, team_name, fetch=None):
    """This club's players and their current refined_availability p_start
    -- the anchor the agent's classifications adjust, never estimate from
    scratch. Same shape predict_gameweek/predict_gameweek_refined return
    (code, web_name, p_start, cold_start, n_observed, method), filtered to
    one club.
    """
    anchor = starts_model.predict_gameweek_refined(
        conn, season, prior_season, target_round, fetch=fetch
    )
    team_by_code = pd.read_sql(
        "SELECT p.code, t.name AS team FROM players p "
        "LEFT JOIN teams t ON t.code = p.team_code",
        conn,
    )
    merged = anchor.merge(team_by_code, on="code", how="left")
    return merged[merged["team"] == team_name].drop(columns=["team"]).reset_index(drop=True)


def run_agent_loop(llm_client, model, team_name, roster, budget,
                    search_web=None, fetch_page_text=None, max_turns=12):
    """The manual ReAct loop -- see module docstring for why it's manual.

    Returns (classifications, fetched_pages): `classifications` is the raw
    list of dicts the model returned via final_answer (unverified --
    verify_classifications does that); `fetched_pages` is {url: text} for
    every page actually fetched during the loop, which verification checks
    quotes against. Returns ([], {}) if the model never reaches a valid
    final_answer within `max_turns` or the tool budget -- the caller treats
    every roster player as no_news in that case, not an error.
    """
    if search_web is None:
        search_web = agent_tools.search_web
    if fetch_page_text is None:
        fetch_page_text = agent_tools.fetch_page_text

    roster_desc = "\n".join(
        "code={0} name={1}".format(row.code, row.web_name)
        for row in roster.itertuples()
    )
    initial = "Club: {0}\nRoster:\n{1}".format(team_name, roster_desc)
    messages = [{"role": "user", "content": initial}]
    fetched_pages = {}

    for _ in range(max_turns):
        response = llm_client.messages.create(
            model=model, system=SYSTEM_PROMPT, max_tokens=1024, messages=messages,
        )
        text = _extract_response_text(response)
        messages.append({"role": "assistant", "content": text})

        try:
            action = json.loads(_strip_json_fences(text))
        except ValueError:
            messages.append({
                "role": "user",
                "content": "That wasn't valid JSON. Respond with exactly one "
                           "JSON action object and nothing else.",
            })
            continue

        kind = action.get("action")
        if kind == "final_answer":
            return action.get("classifications") or [], fetched_pages

        if kind == "search_web":
            try:
                budget.take_search()
                results = search_web(action.get("query", ""))
                observation = json.dumps(results)
            except ToolBudgetExceeded:
                observation = (
                    "Search budget exhausted. Use fetch_page_text on a "
                    "result you already have, or call final_answer now."
                )
            messages.append({"role": "user", "content": observation})
            continue

        if kind == "fetch_page_text":
            url = action.get("url", "")
            try:
                budget.take_fetch()
                page_text = fetch_page_text(url)
                fetched_pages[url] = page_text
                observation = page_text
            except ToolBudgetExceeded:
                observation = (
                    "Fetch budget exhausted. Call final_answer with what "
                    "you have."
                )
            except Exception as exc:  # noqa: BLE001 -- a fetch failure is
                # the agent's problem to route around (try another source
                # or give up on that player), not this loop's to raise.
                observation = "Could not fetch that page: {0}".format(exc)
            messages.append({"role": "user", "content": observation})
            continue

        messages.append({
            "role": "user",
            "content": "Unrecognized action '{0}'. Valid actions are "
                       "search_web, fetch_page_text, final_answer.".format(kind),
        })

    return [], fetched_pages


def verify_classifications(raw_classifications, fetched_pages, valid_codes):
    """Keep only classifications whose quote is an exact substring of the
    page the agent says it came from, and whose code is actually on the
    roster. Discarding on failure here -- not trusting a model's citation
    at face value -- is the non-LLM self-check that catches a citation the
    model paraphrased or invented outright.
    """
    verified = []
    for item in raw_classifications:
        code = item.get("code")
        quote = item.get("quote") or ""
        source_url = item.get("source_url") or ""
        if code not in valid_codes:
            continue
        page_text = fetched_pages.get(source_url)
        if not page_text or quote not in page_text:
            continue
        verified.append(item)
    return verified


def predict_club_agent(conn, season, prior_season, target_round, team_name,
                        search_web=None, fetch_page_text=None, llm_client=None,
                        model=DEFAULT_MODEL, budget=None, fetch=None):
    """P(starts) for one club's roster, agent-adjusted. Returns the same
    shape as predict_gameweek_refined (code, web_name, p_start, cold_start,
    n_observed, method) plus `category`/`quote`/`source_url` audit columns
    (None where the agent found nothing) -- the extra columns are ignored
    by derived._load_predictions but kept in the snapshot JSON for manual
    review.
    """
    roster = load_club_roster(conn, season, prior_season, target_round, team_name, fetch=fetch)
    if len(roster) == 0:
        return roster.assign(category=None, quote=None, source_url=None)

    if llm_client is None:
        import anthropic
        llm_client = anthropic.Anthropic()
    if budget is None:
        budget = ToolBudget()

    raw_classifications, fetched_pages = run_agent_loop(
        llm_client, model, team_name, roster, budget,
        search_web=search_web, fetch_page_text=fetch_page_text,
    )
    verified = verify_classifications(
        raw_classifications, fetched_pages, set(roster["code"])
    )

    category_rates = categories.fit_category_rates(conn, season, target_round)

    result = roster.copy()
    result["category"] = None
    result["quote"] = None
    result["source_url"] = None
    result["method"] = "agent_fallback_no_news"

    by_code = result.set_index("code")
    for item in verified:
        code = item["code"]
        category = item["category"]
        p_start = categories.category_to_p_start(category, category_rates)
        if p_start is None:
            continue
        by_code.loc[code, "p_start"] = p_start
        by_code.loc[code, "method"] = "agent_" + category
        by_code.loc[code, "category"] = category
        by_code.loc[code, "quote"] = item["quote"]
        by_code.loc[code, "source_url"] = item.get("source_url")

    return by_code.reset_index()


def _main():
    import argparse
    import sqlite3

    from fpl_starts import derived

    parser = argparse.ArgumentParser(
        description="Agent-classify one or more clubs' rosters and snapshot "
                    "the result as model_version=refined_availability_agent_news."
    )
    parser.add_argument("--db-path", default=derived.DERIVED_DB_PATH)
    parser.add_argument("--season", default=None,
                        help="Defaults to the only season in derived.db.")
    parser.add_argument("--prior-season", default=None,
                        help="Defaults to one season before --season.")
    parser.add_argument("--target-round", type=int, default=None,
                        help="Defaults to (max archived round) + 1.")
    parser.add_argument("--team", action="append", required=True,
                        help="Club name (as it appears in the teams table), "
                             "repeatable.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--predictions-dir", default=starts_model.PREDICTIONS_DIR)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    season = args.season
    if season is None:
        seasons = pd.read_sql(
            "SELECT DISTINCT season FROM player_gameweek_stats", conn
        )["season"]
        if len(seasons) != 1:
            raise SystemExit(
                "derived.db has {0} seasons; pass --season explicitly".format(len(seasons))
            )
        season = seasons.iloc[0]

    prior_season = args.prior_season
    if prior_season is None:
        start_year = int(season[:4]) - 1
        prior_season = "{0}-{1:02d}".format(start_year, (start_year + 1) % 100)

    target_round = args.target_round
    if target_round is None:
        max_round = pd.read_sql(
            "SELECT MAX(round) AS r FROM player_gameweek_stats WHERE season = ?",
            conn, params=(season,),
        )["r"].iloc[0]
        target_round = int(max_round) + 1 if max_round is not None else 1

    frames = []
    for team_name in args.team:
        print("classifying {0}...".format(team_name))
        frame = predict_club_agent(
            conn, season, prior_season, target_round, team_name, model=args.model,
        )
        n_classified = int((frame["category"].notna()).sum())
        print("  {0}: {1}/{2} players classified".format(
            team_name, n_classified, len(frame)
        ))
        frames.append(frame)
    conn.close()

    combined = pd.concat(frames, ignore_index=True)
    audit_cols = [c for c in ("category", "quote", "source_url") if c in combined.columns]
    snapshot_cols = ["code", "web_name", "p_start", "cold_start", "n_observed", "method"] + audit_cols
    path = starts_model.snapshot_predictions(
        combined[snapshot_cols], season, target_round,
        model_version="refined_availability_agent_news",
        base_dir=args.predictions_dir,
    )
    print("snapshot -> {0}".format(path))


if __name__ == "__main__":
    _main()
