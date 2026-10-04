"""Conversation memory: the agent's earlier turns come from a LangGraph
checkpointer keyed by thread ID, never from resending the chat transcript.
A fake chat model records what it's sent -- no API calls."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
import pytest

import agent


class Recording(BaseChatModel):
    """Answers with the next of `replies` (plain text, no tool calls) and
    records the messages it was sent on each call."""
    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "recording-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.replies.pop(0)))])


@pytest.fixture
def model(monkeypatch):
    """The fake model build_agent will use; set `.replies` before invoking."""
    fake = Recording()
    monkeypatch.setattr(agent, "build_chat_model", lambda config=None: fake)
    return fake


def _ask(graph, question, thread_id=None):
    config = agent.thread_config(thread_id) if thread_id else None
    return graph.invoke({"messages": [{"role": "user", "content": question}]}, config=config)


def _conversation(call):
    """(type, text) of each message the model was sent, without the system prompt."""
    return [(m.type, m.content) for m in call if m.type != "system"]


def test_the_same_thread_sees_the_earlier_turns(model):
    model.replies = ["One player is at risk. Want me to find replacements?", "Here are some replacements."]
    graph = agent.build_agent(checkpointer=InMemorySaver())
    _ask(graph, "Who is at risk?", "thread-A")
    _ask(graph, "yes", "thread-A")  # only the new message is sent
    assert _conversation(model.calls[1]) == [
        ("human", "Who is at risk?"),
        ("ai", "One player is at risk. Want me to find replacements?"),
        ("human", "yes"),
    ]


def test_different_threads_are_isolated(model):
    model.replies = ["Saka is 40% likely to start.", "Hello."]
    graph = agent.build_agent(checkpointer=InMemorySaver())
    _ask(graph, "Tell me about Saka.", "thread-A")
    _ask(graph, "What did I just ask?", "thread-B")
    assert _conversation(model.calls[1]) == [("human", "What did I just ask?")]


def test_a_new_thread_id_starts_a_fresh_conversation(model):
    """What the app's New chat does: same graph and saver, new thread ID."""
    model.replies = ["First answer.", "Second answer.", "Fresh answer."]
    graph = agent.build_agent(checkpointer=InMemorySaver())
    _ask(graph, "First question", "old-thread")
    _ask(graph, "Second question", "old-thread")
    _ask(graph, "New question", "new-thread")
    assert _conversation(model.calls[2]) == [("human", "New question")]
    old = graph.get_state(agent.thread_config("old-thread")).values["messages"]
    assert [m.content for m in old] == ["First question", "First answer.", "Second question", "Second answer."]


def test_without_a_checkpointer_every_invoke_starts_from_nothing(model):
    """The CLI and the evals build the agent without one, as before."""
    model.replies = ["Want me to find replacements?", "Sorry, yes to what?"]
    graph = agent.build_agent()
    _ask(graph, "Who is at risk?")
    _ask(graph, "yes")
    assert _conversation(model.calls[1]) == [("human", "yes")]


def test_the_thread_config_keys_the_checkpointer_and_the_trace_metadata():
    config = agent.thread_config("abc-123", app="fpl-starts", season="2026-27")
    assert config == {"configurable": {"thread_id": "abc-123"},
                      "metadata": {"thread_id": "abc-123", "app": "fpl-starts", "season": "2026-27"}}


class MetadataSeen(BaseCallbackHandler):
    """The metadata each run (graph, nodes, model) is started with -- what
    LangSmith's tracer receives for every run in the trace tree."""

    def __init__(self):
        self.runs = []

    def on_chain_start(self, serialized, inputs, *, metadata=None, **kwargs):
        self.runs.append(("chain", dict(metadata or {})))

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        self.runs.append(("model", dict(metadata or {})))


def test_the_thread_metadata_reaches_every_run_in_the_trace(model):
    model.replies = ["Answer."]
    graph = agent.build_agent(checkpointer=InMemorySaver())
    seen = MetadataSeen()
    config = agent.thread_config("abc-123", app="fpl-starts")
    config["callbacks"] = [seen]
    graph.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config=config)
    assert any(kind == "model" for kind, _ in seen.runs)
    assert all(m.get("thread_id") == "abc-123" and m.get("app") == "fpl-starts" for _, m in seen.runs)
