"""The chat model behind the agent -- the one place that knows which
provider supplies it.

The agent needs a LangChain chat model that takes a system prompt and
messages, binds tools, returns structured tool calls and continues from
tool results (LangGraph's create_react_agent does the rest). Anthropic and
OpenAI both provide one; which is used is configuration, not code:

    LLM_PROVIDER=anthropic            # or openai; default anthropic
    ANTHROPIC_MODEL=claude-opus-5     # default for anthropic
    OPENAI_MODEL=gpt-6-sol            # default for openai
    LLM_MAX_TOKENS=8000               # default 8000

plus the selected provider's key (ANTHROPIC_API_KEY or OPENAI_API_KEY) and,
for an Anthropic key not scoped to one workspace, ANTHROPIC_WORKSPACE_ID.
Only the selected provider's settings are read or checked; an unknown
provider or a missing key is an error, never a quiet switch to the other
provider.

Provider-specific options (headers, reasoning controls) belong in the
provider's builder below, not in LLMConfig.

These settings (and the rest of the app's, e.g. LANGSMITH_*) are loaded
from the repo-root .env with python-dotenv when this module is imported;
variables already set in the environment win over the file.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MAX_TOKENS = 8000

# provider -> (model setting, default model, API key setting)
PROVIDERS = {
    "anthropic": ("ANTHROPIC_MODEL", "claude-opus-5", "ANTHROPIC_API_KEY"),
    "openai": ("OPENAI_MODEL", "gpt-6-sol", "OPENAI_API_KEY"),
}


class LLMConfigError(ValueError):
    """The configured provider can't be used: unknown, or missing its key."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS

    @property
    def label(self):
        """provider:model, e.g. anthropic:claude-opus-5 -- for logs and reports."""
        return "{0}:{1}".format(self.provider, self.model)


def load_llm_config(provider_override=None, model_override=None, env=None):
    """The configured provider and model from the environment (`env`, default
    os.environ), with optional overrides -- the evals' --provider/--model.
    The model defaults to the provider's documented default when its
    *_MODEL setting is unset."""
    env = os.environ if env is None else env
    provider = (provider_override or env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        raise LLMConfigError("Unknown LLM provider {0!r} -- use one of: {1}.".format(
            provider, ", ".join(sorted(PROVIDERS))))
    model_setting, default_model, _ = PROVIDERS[provider]
    model = (model_override or env.get(model_setting) or default_model).strip()
    raw_max = env.get("LLM_MAX_TOKENS") or str(DEFAULT_MAX_TOKENS)
    try:
        max_tokens = int(raw_max)
    except ValueError:
        raise LLMConfigError("LLM_MAX_TOKENS must be a whole number, not {0!r}.".format(raw_max)) from None
    return LLMConfig(provider=provider, model=model, max_tokens=max_tokens)


def missing_credentials(config, env=None):
    """The selected provider's unset key setting, or None. Never reads or
    checks the other provider's key."""
    env = os.environ if env is None else env
    key_setting = PROVIDERS[config.provider][2]
    return None if env.get(key_setting) else key_setting


def build_chat_model(config=None):
    """A LangChain chat model for `config` (default: load_llm_config()).
    Fails clearly if the provider's key is missing or the integration can't
    call tools -- the agent can't work without either."""
    config = config or load_llm_config()
    missing = missing_credentials(config)
    if missing:
        raise LLMConfigError("LLM provider {0!r} is selected but {1} is not configured.".format(
            config.provider, missing))
    model = _BUILDERS[config.provider](config)
    if type(model).bind_tools is BaseChatModel.bind_tools:  # the base class only raises NotImplementedError
        raise LLMConfigError("{0} doesn't support tool calling, which the agent needs.".format(config.label))
    return model


def _build_anthropic(config):
    from langchain_anthropic import ChatAnthropic

    # An API key that isn't scoped to a single workspace needs this header on
    # every request (confirmed live -- omitting it 400s), a key that *is*
    # scoped doesn't need or accept it being wrong, so only send it when set.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return ChatAnthropic(model=config.model, max_tokens=config.max_tokens, default_headers=default_headers)


def _build_openai(config):
    from langchain_openai import ChatOpenAI

    # max_tokens is sent as OpenAI's max_completion_tokens; the key comes
    # from OPENAI_API_KEY in the environment.
    return ChatOpenAI(model=config.model, max_tokens=config.max_tokens)


_BUILDERS = {"anthropic": _build_anthropic, "openai": _build_openai}
