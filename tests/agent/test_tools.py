"""Tests for agent/tools.py: fake HTTP sessions throughout, no real network."""

import pytest

from fpl_starts.agent.tools import (
    ToolBudget,
    ToolBudgetExceeded,
    fetch_page_text,
    html_to_text,
    search_web,
)


class FakeResponse(object):
    def __init__(self, json_data=None, text=""):
        self._json = json_data
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeSession(object):
    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._response


def test_search_web_parses_brave_response():
    session = FakeSession(FakeResponse(json_data={
        "web": {"results": [
            {"title": "Leeds team news", "url": "https://a", "description": "snippet a"},
        ]}
    }))
    results = search_web("Leeds team news", api_key="x", session=session)
    assert results == [{"title": "Leeds team news", "url": "https://a", "snippet": "snippet a"}]
    assert session.calls[0][1]["headers"]["X-Subscription-Token"] == "x"


def test_search_web_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        search_web("query", api_key=None, session=FakeSession(FakeResponse()))


def test_html_to_text_strips_script_and_style():
    html = "<html><head><style>.x{}</style></head><body><script>1</script><p>Hello world</p></body></html>"
    assert html_to_text(html) == "Hello world"


def test_fetch_page_text_truncates_to_max_chars():
    session = FakeSession(FakeResponse(text="<p>" + "x" * 100 + "</p>"))
    text = fetch_page_text("https://a", session=session, max_chars=10)
    assert text == "x" * 10


def test_tool_budget_exhausts_searches_then_raises():
    budget = ToolBudget(max_searches=2, max_fetches=1)
    budget.take_search()
    budget.take_search()
    with pytest.raises(ToolBudgetExceeded):
        budget.take_search()


def test_tool_budget_tracks_search_and_fetch_independently():
    budget = ToolBudget(max_searches=1, max_fetches=1)
    budget.take_search()
    budget.take_fetch()
    with pytest.raises(ToolBudgetExceeded):
        budget.take_search()
    with pytest.raises(ToolBudgetExceeded):
        budget.take_fetch()
