"""The provider boundary (llm.py) and provider-neutral text extraction. The
provider classes are replaced by recorders -- no API calls."""

import langchain_anthropic
import langchain_openai
from langchain_core.language_models.chat_models import BaseChatModel
import pytest

import agent
import llm


# --- configuration -----------------------------------------------------------------------

def test_anthropic_config():
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_MODEL": "claude-opus-5"}
    assert llm.load_llm_config(env=env) == llm.LLMConfig("anthropic", "claude-opus-5", 8000)


def test_openai_config():
    env = {"LLM_PROVIDER": "OpenAI", "OPENAI_MODEL": "gpt-6-sol", "LLM_MAX_TOKENS": "4000"}
    assert llm.load_llm_config(env=env) == llm.LLMConfig("openai", "gpt-6-sol", 4000)


def test_unknown_provider_fails_clearly():
    with pytest.raises(llm.LLMConfigError, match="Unknown LLM provider 'made-up'"):
        llm.load_llm_config(env={"LLM_PROVIDER": "made-up"})


def test_unset_settings_use_the_documented_defaults():
    """Policy: no LLM_PROVIDER means anthropic; no *_MODEL means that
    provider's default -- so an existing .env keeps working unchanged."""
    assert llm.load_llm_config(env={}) == llm.LLMConfig("anthropic", "claude-opus-5", 8000)
    assert llm.load_llm_config(env={"LLM_PROVIDER": "openai"}).model == "gpt-6-sol"


def test_only_the_selected_providers_model_setting_is_read():
    env = {"LLM_PROVIDER": "openai", "ANTHROPIC_MODEL": "claude-x", "OPENAI_MODEL": "gpt-y"}
    assert llm.load_llm_config(env=env).model == "gpt-y"


def test_overrides_win_over_the_environment():
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_MODEL": "claude-opus-5"}
    assert llm.load_llm_config("openai", "gpt-z", env=env) == llm.LLMConfig("openai", "gpt-z", 8000)


def test_a_bad_max_tokens_fails_clearly():
    with pytest.raises(llm.LLMConfigError, match="LLM_MAX_TOKENS"):
        llm.load_llm_config(env={"LLM_MAX_TOKENS": "lots"})


# --- credentials ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider, env, missing", [
    ("anthropic", {"ANTHROPIC_API_KEY": "a"}, None),
    ("openai", {"OPENAI_API_KEY": "o"}, None),
    ("openai", {"ANTHROPIC_API_KEY": "a"}, "OPENAI_API_KEY"),
    ("anthropic", {"OPENAI_API_KEY": "o"}, "ANTHROPIC_API_KEY"),
])
def test_only_the_selected_providers_key_is_required(provider, env, missing):
    assert llm.missing_credentials(llm.LLMConfig(provider, "m"), env=env) == missing


def test_a_missing_key_stops_the_build_and_names_it(monkeypatch, recorders):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    with pytest.raises(llm.LLMConfigError, match="'openai' is selected but OPENAI_API_KEY is not configured"):
        llm.build_chat_model(llm.LLMConfig("openai", "gpt-6-sol"))
    assert recorders == []  # no client built, and no fallback to Anthropic


# --- factory -------------------------------------------------------------------------------

@pytest.fixture
def recorders(monkeypatch):
    """Replace both provider classes; returns the (provider, kwargs) each
    construction was called with."""
    built = []

    def recorder(name):
        class Recorder:
            def __init__(self, **kwargs):
                built.append((name, kwargs))

            def bind_tools(self, tools, **kwargs):
                return self
        return Recorder

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", recorder("anthropic"))
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", recorder("openai"))
    return built


def test_anthropic_config_builds_chatanthropic_only(monkeypatch, recorders):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_1")
    llm.build_chat_model(llm.LLMConfig("anthropic", "claude-opus-5", 8000))
    assert recorders == [("anthropic", {"model": "claude-opus-5", "max_tokens": 8000,
                                         "default_headers": {"anthropic-workspace-id": "wrkspc_1"}})]


def test_openai_config_builds_chatopenai_only_without_anthropic_headers(monkeypatch, recorders):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_1")  # set, but Anthropic-only
    llm.build_chat_model(llm.LLMConfig("openai", "gpt-6-sol", 8000))
    assert recorders == [("openai", {"model": "gpt-6-sol", "max_tokens": 8000})]


def test_no_workspace_header_unless_configured(monkeypatch, recorders):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    llm.build_chat_model(llm.LLMConfig("anthropic", "claude-opus-5"))
    assert recorders[0][1]["default_headers"] is None


def test_a_model_without_tool_calling_is_refused(monkeypatch):
    class NoTools(BaseChatModel):
        @property
        def _llm_type(self):
            return "no-tools"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise AssertionError("never called")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setitem(llm._BUILDERS, "anthropic", lambda config: NoTools())
    with pytest.raises(llm.LLMConfigError, match="doesn't support tool calling"):
        llm.build_chat_model(llm.LLMConfig("anthropic", "claude-opus-5"))


def test_the_agent_gets_its_model_from_the_factory(monkeypatch):
    """build_agent passes its llm_config straight to the factory."""
    seen = []
    monkeypatch.setattr(agent, "build_chat_model", lambda config=None: seen.append(config) or _Fake())
    config = llm.LLMConfig("openai", "gpt-6-sol")
    agent.build_agent(llm_config=config)
    assert seen == [config]


class _Fake(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("never called")


# --- extract_text --------------------------------------------------------------------------

def test_plain_string_content():
    assert agent.extract_text("Saka is 40% likely to start.") == "Saka is 40% likely to start."


def test_anthropic_thinking_blocks_are_dropped():
    content = [{"type": "thinking", "thinking": "hmm", "signature": "abc"},
               {"type": "text", "text": "Haaland is 97%."}]
    assert agent.extract_text(content) == "Haaland is 97%."


def test_openai_reasoning_blocks_are_dropped():
    content = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret"}]},
               {"type": "text", "text": "Rice is "}, {"type": "output_text", "text": "93%."}]
    assert agent.extract_text(content) == "Rice is 93%."


def test_tool_use_blocks_and_bare_strings():
    content = ["Looking that up. ", {"type": "tool_use", "name": "explain_player", "input": {}}]
    assert agent.extract_text(content) == "Looking that up. "
    assert agent.extract_text([]) == ""
