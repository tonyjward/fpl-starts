"""The agent's two tools: search the web, and fetch/extract one page's text.

Both are plain functions the agent's loop (predict.py) calls directly --
there is no framework mediating this, per the module docstring in
predict.py. Both are budget-capped by `ToolBudget` so a bad run degrades to
"fewer players classified", never an unbounded loop or runaway API cost.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import os

import requests
from bs4 import BeautifulSoup

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Page text is truncated to this many characters before being handed back to
# the agent -- long enough for a team-news article, short enough not to
# blow the context budget across a multi-tool-call loop.
MAX_PAGE_CHARS = 4000


class ToolBudgetExceeded(Exception):
    """A tool was called after its per-club budget was already spent."""


class ToolBudget(object):
    """Tracks remaining search/fetch calls for one club's agent run.

    Enforced in code, not left to the model's discretion -- see predict.py's
    module docstring for why.
    """

    def __init__(self, max_searches=3, max_fetches=2):
        self.searches_remaining = max_searches
        self.fetches_remaining = max_fetches

    def take_search(self):
        if self.searches_remaining <= 0:
            raise ToolBudgetExceeded("search budget exhausted")
        self.searches_remaining -= 1

    def take_fetch(self):
        if self.fetches_remaining <= 0:
            raise ToolBudgetExceeded("fetch budget exhausted")
        self.fetches_remaining -= 1


def search_web(query, api_key=None, session=None):
    """Brave Search API: {query} -> list of {title, url, snippet}.

    `session` is an injectable requests.Session-like object (needs a `.get`
    matching requests' signature/response shape) -- real network calls only
    happen when this isn't overridden, same fake-injection pattern as
    `fetch` throughout the rest of this project (e.g.
    starts_model.fetch_community_archive).
    """
    if api_key is None:
        api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY not set")
    if session is None:
        session = requests
    resp = session.get(
        BRAVE_SEARCH_URL,
        params={"q": query, "count": 10},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = []
    for item in (payload.get("web") or {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    return results


def html_to_text(html):
    """Visible text from an HTML page, script/style stripped, collapsed
    whitespace -- good enough for an LLM to read a news article from, not a
    faithful rendering.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def fetch_page_text(url, session=None, max_chars=MAX_PAGE_CHARS):
    """GET `url` and return its extracted, truncated text.

    Same injectable-session pattern as search_web.
    """
    if session is None:
        session = requests
    resp = session.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (compatible; fpl-starts-agent/0.1)",
    })
    resp.raise_for_status()
    text = html_to_text(resp.text)
    return text[:max_chars]
