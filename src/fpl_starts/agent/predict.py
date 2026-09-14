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

# Methods on the anchor (refined_availability's own output) that represent
# an actual FPL-fitness-based decision -- a news/agent classification must
# never override these, same precedence rule the private repo's own
# route_predictions_with_news uses (`decided_codes`). confirmed_out is the
# one deliberate exception (see predict_club_agent): it still applies on
# top of an existing decided row, since 0.0 is either consistent with an
# existing gate or catches a fresh claim FPL's own flag hasn't updated for
# yet -- everything else defers to the fitness assessment already made.
_AVAILABILITY_DECIDED_METHODS = frozenset(
    ["hard_gate_unavailable", "flag_table", "flag_table_pooled"]
)

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

You will be told which fixture (opponent club) you are gathering evidence \
for. Every claim must be about *that* fixture specifically -- articles get \
reused or resurface from a different game, so classify a claim only if the \
article is clearly about the fixture you were given, not an old or \
unrelated one.

Every classification also needs a content_type, classified honestly \
regardless of what category you picked: "team_news" if the article is \
forward-looking (previewing the upcoming match), or "match_report" if it \
describes a match that has already been played (e.g. reporting a player \
was substituted or sent off in a *previous* game -- that is not evidence \
about the *next* one, even if it looks like it at a glance).

Your category must match what your own quote actually says. Use \
confirmed_out or confirmed_starting only when the evidence is unhedged --  \
words like "doubtful", "assessed", "50-50", or "a fitness test" describe \
uncertainty, not a confirmation, even if the same sentence also lists a \
player who genuinely is confirmed out. Use rotation_risk or \
returning_from_injury for anything short of that.

A player may appear more than once in your final classifications if your \
sources genuinely disagree about them -- report every distinct piece of \
evidence you found rather than picking one to report. Each occurrence \
needs its own category, quote, and source_url, exactly like a single \
classification would.

Respond with exactly one JSON object per turn, and nothing else -- no \
markdown fences, no prose outside the JSON. Valid actions:

{{"action": "search_web", "query": "..."}}
{{"action": "fetch_page_text", "url": "..."}}
{{"action": "final_answer", "classifications": [
  {{"code": <player code, integer>, "category": "<one of the categories above>", \
"content_type": "team_news or match_report", \
"opponent": "<the opponent club the article is about, or an empty string if none is named>", \
"quote": "<verbatim quote from a fetched page>", "source_url": "<the url that quote came from>"}}
]}}

Every quote must be copied verbatim from a page you fetched with \
fetch_page_text -- not paraphrased, not from a search snippet alone. \
Omit any player you found no evidence for; they are handled separately. \
Call final_answer once you've covered the roster or have used your \
available tool calls.""".format(taxonomy=TAXONOMY_DESCRIPTIONS)


# Curly/smart quote variants mapped to their plain-ASCII equivalents --
# confirmed as a real bug in the private repo's own news pipeline: a source
# article using curly quotes made a straight-quote substring check fail on
# an otherwise byte-correct citation, silently rejecting every claim from
# that article. Not a fabrication risk to normalize away: the words
# themselves are unchanged, only the character variant.
_QUOTE_CHAR_MAP = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
}


def _normalize_for_substring_check(text):
    """Lowercased, whitespace-collapsed, quote-character-normalized form of
    `text`, used only to compare a claimed quote against the page it's
    supposed to have come from -- not stored anywhere, the classification's
    own `quote` field keeps the original text for display/audit.
    """
    text = text or ""
    for curly, straight in _QUOTE_CHAR_MAP.items():
        text = text.replace(curly, straight)
    return " ".join(text.lower().split())


# Deliberately lexical/deterministic, not another LLM call -- same "verify
# in code" discipline as quote verification itself. Flags a classification
# as suspect only for the two categories where a false confirmation is most
# costly (confirmed_out hard-gates to 0; confirmed_starting carries the
# highest prior) -- hedging language in a rotation_risk/returning_from_injury
# quote is exactly what that category is for, not a red flag.
HEDGE_PHRASES = (
    "doubtful", "questionable", "assessed", "50-50", "50/50",
    "game-time decision", "monitored", "touch and go", "fitness test",
    "waiting on", "could return", "may be", "might be", "not yet confirmed",
)


def _hedge_phrases_in(quote):
    normalized = _normalize_for_substring_check(quote)
    return any(phrase in normalized for phrase in HEDGE_PHRASES)


def _inconsistent_classifications(classifications):
    """Classifications whose category (confirmed_out/confirmed_starting)
    contradicts hedging language in their own quote -- see HEDGE_PHRASES.
    Used both to trigger a self-correction turn in run_agent_loop and, as a
    final safety net, in verify_classifications.
    """
    return [
        item for item in classifications
        if item.get("category") in ("confirmed_out", "confirmed_starting")
        and _hedge_phrases_in(item.get("quote") or "")
    ]


def get_opponent(conn, season, target_round, team_name):
    """The opponent club name for `team_name`'s fixture in `target_round`,
    or None if no fixture is recorded (e.g. a blank gameweek). Used both to
    tell the agent which fixture it's gathering evidence for and to verify
    a returned `opponent` field isn't about some other game.
    """
    row = pd.read_sql(
        "SELECT t2.name AS opponent FROM fixtures f "
        "JOIN teams t1 ON t1.code = f.team_code "
        "JOIN teams t2 ON t2.code = f.opponent_code "
        "WHERE f.season = ? AND f.round = ? AND t1.name = ?",
        conn, params=(season, target_round, team_name),
    )
    if len(row) == 0:
        return None
    return row["opponent"].iloc[0]


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


def run_agent_loop(llm_client, model, team_name, roster, budget, opponent_name=None,
                    search_web=None, fetch_page_text=None, max_turns=12,
                    max_reclassifications=1):
    """The manual ReAct loop -- see module docstring for why it's manual.

    `opponent_name` (from get_opponent) tells the model which fixture it's
    gathering evidence for, so it can tell a genuinely current article from
    a stale or reused one about a different game -- None if no fixture is
    known for this round (the model is told evidence can't be fixture-
    checked, which the caller should treat as a reason to be more, not
    less, skeptical of what comes back).

    A `final_answer` isn't accepted immediately: any confirmed_out/
    confirmed_starting classification whose own quote contains hedging
    language (see HEDGE_PHRASES) gets one corrective turn -- named
    specifically, so the model can fix its own mistake rather than the
    evidence it already gathered being thrown away -- capped at
    `max_reclassifications` rounds so this is a correction opportunity, not
    an invitation to loop indefinitely second-guessing itself. Whatever
    comes back after that (fixed or not) is returned as-is;
    verify_classifications is the final safety net if it's still wrong.

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
    if opponent_name:
        fixture_line = "Next fixture: {0} vs {1}\n".format(team_name, opponent_name)
    else:
        fixture_line = (
            "Next fixture: unknown -- no fixture is recorded for this round, "
            "so you cannot fixture-check any claim you find. Be more "
            "cautious about staleness as a result.\n"
        )
    initial = "Club: {0}\n{1}Roster:\n{2}".format(team_name, fixture_line, roster_desc)
    messages = [{"role": "user", "content": initial}]
    fetched_pages = {}
    reclassifications_used = 0

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
            classifications = action.get("classifications") or []
            inconsistent = _inconsistent_classifications(classifications)
            if inconsistent and reclassifications_used < max_reclassifications:
                reclassifications_used += 1
                issues = "; ".join(
                    "code={0} category={1} quote={2!r} reads as hedged, not "
                    "confirmed".format(item.get("code"), item.get("category"), item.get("quote"))
                    for item in inconsistent
                )
                messages.append({
                    "role": "user",
                    "content": "Some of your classifications don't match their own "
                               "quotes: {0}. Re-read them and resubmit a corrected "
                               "final_answer -- use rotation_risk or "
                               "returning_from_injury where the evidence is hedged, "
                               "not confirmed_out/confirmed_starting.".format(issues),
                })
                continue
            return classifications, fetched_pages

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


def verify_classifications(raw_classifications, fetched_pages, valid_codes, opponent_name=None):
    """Keep only classifications whose quote is an exact (normalized)
    substring of the page the agent says it came from, whose code is
    actually on the roster, whose content_type is forward-looking (not a
    report of an already-played match), and whose named opponent -- if any
    -- matches the fixture actually being predicted for.

    Discarding on failure here -- not trusting a model's citation or
    self-classification at face value -- is the non-LLM self-check that
    catches a citation the model paraphrased or invented outright, or a
    stale/wrong-fixture article it didn't recognize as such. `opponent_name`
    of None (no fixture on record for this round -- see run_agent_loop)
    means the opponent check is skipped entirely, same as an empty claimed
    opponent: absence of a check is never itself a mismatch.
    """
    verified = []
    for item in raw_classifications:
        code = item.get("code")
        quote = item.get("quote") or ""
        source_url = item.get("source_url") or ""
        content_type = item.get("content_type") or "team_news"
        claimed_opponent = (item.get("opponent") or "").strip()

        if code not in valid_codes:
            continue
        if content_type == "match_report":
            continue
        if opponent_name and claimed_opponent and claimed_opponent.lower() != opponent_name.lower():
            continue
        page_text = fetched_pages.get(source_url)
        if not page_text:
            continue
        if _normalize_for_substring_check(quote) not in _normalize_for_substring_check(page_text):
            continue
        verified.append(item)
    # Final safety net: run_agent_loop already gives the model one chance to
    # fix a confirmed_out/confirmed_starting classification whose own quote
    # is hedged (see HEDGE_PHRASES), but if it didn't take that chance (or
    # ignored the correction), don't trust it here either.
    inconsistent_codes = {item["code"] for item in _inconsistent_classifications(verified)}
    return [item for item in verified if item["code"] not in inconsistent_codes]


def _blend_verified(items, category_rates):
    """Blend one player's verified classifications into a single p_start,
    weighted by claim count per category -- mirrors the private repo's
    route_predictions_with_news, reimplemented here (not imported).

    Returns (p_start, method, forced), or None if nothing in `items` is
    priceable (the caller treats that the same as no evidence at all).

    `forced=True` only for a unanimous confirmed_out set -- applies
    regardless of any existing FPL-status decision (see
    predict_club_agent), same as the private repo's own ordering (the
    confirmed_out check happens before the availability-precedence check).
    Otherwise `forced=False`: the mean of each priced item's
    categories.category_to_p_start value (confirmed_out itself already
    prices at exactly 0.0, so a *non-unanimous* set that includes it
    correctly pulls the average down rather than being treated as
    certain). `method` is "agent_<category>" when every priced item shares
    one category, else "agent_blended".

    Duplicate items (same quote after _normalize_for_substring_check) are
    collapsed to one before weighting first -- syndicated/duplicate
    content must not count twice toward the blend, same reasoning as the
    private repo's own dedupe_claims fix.
    """
    deduped = []
    seen_quotes = set()
    for item in items:
        key = _normalize_for_substring_check(item.get("quote") or "")
        if key in seen_quotes:
            continue
        seen_quotes.add(key)
        deduped.append(item)

    categories_present = {item["category"] for item in deduped}
    if categories_present == {"confirmed_out"}:
        return 0.0, "agent_confirmed_out", True

    priced = []
    for item in deduped:
        p = categories.category_to_p_start(item["category"], category_rates)
        if p is not None:
            priced.append((item["category"], p))
    if not priced:
        return None

    p_start = sum(p for _, p in priced) / len(priced)
    priced_categories = {category for category, _ in priced}
    method = (
        "agent_" + next(iter(priced_categories))
        if len(priced_categories) == 1 else "agent_blended"
    )
    return p_start, method, False


def predict_club_agent(conn, season, prior_season, target_round, team_name,
                        search_web=None, fetch_page_text=None, llm_client=None,
                        model=DEFAULT_MODEL, budget=None, fetch=None):
    """P(starts) for one club's roster, agent-adjusted. Returns the same
    shape as predict_gameweek_refined (code, web_name, p_start, cold_start,
    n_observed, method) plus an `evidence` audit column -- a JSON-encoded
    list of `{category, quote, source_url}` dicts (one entry per distinct
    classification contributing to this player's blend; None where the
    agent found nothing at all). Ignored by derived._load_predictions but
    kept in the snapshot JSON for manual review and (see
    domain_stats.rebuild_evidence_table) per-source accuracy tracking.
    """
    roster = load_club_roster(conn, season, prior_season, target_round, team_name, fetch=fetch)
    if len(roster) == 0:
        return roster.assign(evidence=None)

    if llm_client is None:
        import anthropic
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        llm_client = anthropic.Anthropic(default_headers=headers)
    if budget is None:
        budget = ToolBudget()

    opponent_name = get_opponent(conn, season, target_round, team_name)

    raw_classifications, fetched_pages = run_agent_loop(
        llm_client, model, team_name, roster, budget, opponent_name=opponent_name,
        search_web=search_web, fetch_page_text=fetch_page_text,
    )
    verified = verify_classifications(
        raw_classifications, fetched_pages, set(roster["code"]), opponent_name=opponent_name,
    )

    category_rates = categories.fit_category_rates(conn, season, target_round)
    anchor_method_by_code = roster.set_index("code")["method"]

    result = roster.copy()
    result["evidence"] = None
    result["method"] = "agent_fallback_no_news"
    by_code = result.set_index("code")

    verified_by_code = {}
    for item in verified:
        verified_by_code.setdefault(item["code"], []).append(item)

    for code, items in verified_by_code.items():
        blended = _blend_verified(items, category_rates)
        if blended is None:
            continue
        p_start, method, forced = blended
        evidence_json = json.dumps([
            {"category": item["category"], "quote": item["quote"],
             "source_url": item.get("source_url")}
            for item in items
        ])

        if not forced and anchor_method_by_code.get(code) in _AVAILABILITY_DECIDED_METHODS:
            # FPL status already made a real fitness-based call for this
            # player (injured/suspended/doubtful) -- a lower-confidence
            # news classification doesn't get to override it (unanimous
            # confirmed_out is the deliberate exception -- see
            # _blend_verified). p_start stays at the anchor's own value
            # (already copied into `result`); the evidence column and a
            # dedicated method label still record what the agent found,
            # so it's clear it deferred rather than found nothing at all.
            by_code.loc[code, "method"] = "agent_deferred_to_availability"
            by_code.loc[code, "evidence"] = evidence_json
            continue

        by_code.loc[code, "p_start"] = p_start
        by_code.loc[code, "method"] = method
        by_code.loc[code, "evidence"] = evidence_json

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
        n_classified = int((frame["evidence"].notna()).sum())
        print("  {0}: {1}/{2} players classified".format(
            team_name, n_classified, len(frame)
        ))
        frames.append(frame)
    conn.close()

    combined = pd.concat(frames, ignore_index=True)
    audit_cols = ["evidence"] if "evidence" in combined.columns else []
    snapshot_cols = ["code", "web_name", "p_start", "cold_start", "n_observed", "method"] + audit_cols
    path = starts_model.snapshot_predictions(
        combined[snapshot_cols], season, target_round,
        model_version="refined_availability_agent_news",
        base_dir=args.predictions_dir,
    )
    print("snapshot -> {0}".format(path))


if __name__ == "__main__":
    _main()
