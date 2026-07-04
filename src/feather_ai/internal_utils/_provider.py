import os
import re
from typing import Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models.chat_models import BaseChatModel
from ._exceptions import ModelNotSupportedException, ApiKeyMissingException

provider_mapping = {
    "gemini": ChatGoogleGenerativeAI,
    "claude": ChatAnthropic,
    "openai": ChatOpenAI,
    "mistral": ChatMistralAI,
    "deepseek": ChatDeepSeek,
}

model_mapping = {
    "gemini": lambda model: model.startswith("gemini") or model.startswith("gemma"),
    "claude": lambda model: model.startswith("claude"),
    "openai": lambda model: model.startswith("gpt"),
    "mistral": lambda model: model.startswith("mistral"),
    "deepseek": lambda model: model.startswith("deepseek"),
}

env_vars = {
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Matches a trailing `-thinking-<level>` suffix on gemini model names, e.g.
# "gemini-3.5-flash-thinking-minimal" -> base "gemini-3.5-flash", level "minimal".
_THINKING_SUFFIX_RE = re.compile(r"-thinking-(minimal|low|medium|high)$")


def parse_thinking_suffix(model: str) -> Tuple[str, Optional[str]]:
    """Split a trailing ``-thinking-<level>`` suffix off a gemini model string.

    Model strings such as ``gemini-3.5-flash-thinking-minimal`` encode a Gemini
    ``thinking_level`` (a Gemini 3+ API feature exposed by
    langchain-google-genai>=4.2.6 as ``thinking_level: Literal["minimal","low",
    "medium","high"] | None``). This helper strips that suffix so the real model
    name can be passed to ``ChatGoogleGenerativeAI`` while the parsed level is
    surfaced separately.

    Args:
        model: the (possibly suffixed) model name.

    Returns:
        ``(base_model, level)`` where ``base_model`` has the suffix removed and
        ``level`` is one of ``minimal|low|medium|high``. Returns ``(model, None)``
        when there is no valid suffix or when the model is not a gemini model.

    Note:
        Parsing is intentionally not gated on the Gemini version — e.g.
        ``gemini-2.5-flash-thinking-high`` still parses. The Gemini API itself
        rejects ``thinking_level`` on pre-3 models, so no version check is done
        here.
    """
    if not model.startswith("gemini"):
        return model, None
    match = _THINKING_SUFFIX_RE.search(model)
    if match is None:
        return model, None
    return model[: match.start()], match.group(1)


def get_provider(model: str) -> Tuple[BaseChatModel, str]:
    """
    get the specified LLM provider baseclass
    Args:
        model: string containing model name

    Returns:
    The Chat baseclass from langchain
    """
    # Strip an optional `-thinking-<level>` suffix off gemini model names so the
    # real model name is used for provider detection and passed to the provider.
    base_model, thinking_level = parse_thinking_suffix(model)

    provider_key = None
    # Extract the provider prefix from the model name
    for key in provider_mapping.keys():
        if model_mapping[key](base_model):
            provider_key = key
            break

    # Error handling if model does not exist
    if provider_key is None:
        raise ModelNotSupportedException(model)

    if not os.getenv(env_vars[provider_key]):
        raise ApiKeyMissingException(provider_key, env_vars[provider_key])

    model_kwargs = {
        "model": base_model,
        "streaming": True,
    }
    # OpenAI-compatible chat models only attach token usage to streamed chunks
    # when explicitly asked to. Gemini/Anthropic/Mistral attach it natively.
    if provider_key in ("openai", "deepseek"):
        model_kwargs["stream_usage"] = True

    # Only gemini supports thinking_level, and only when a level was parsed.
    if provider_key == "gemini" and thinking_level is not None:
        model_kwargs["thinking_level"] = thinking_level

    return provider_mapping[provider_key](
        **model_kwargs,  # type: ignore
    ), provider_key