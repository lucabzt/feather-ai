import os
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from ._exceptions import ModelNotSupportedException, ApiKeyMissingException

provider_mapping = {
    "gemini": ChatGoogleGenerativeAI,
    "claude": ChatAnthropic,
    "openai": ChatOpenAI,
    "mistral": ChatMistralAI
}

model_mapping = {
    "gemini": lambda model: model.startswith("gemini"),
    "claude": lambda model: model.startswith("claude"),
    "openai": lambda model: model.startswith("gpt"),
    "mistral": lambda model: model.startswith("mistral")
}

env_vars = {
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY"
}

def get_provider(model: str) -> Tuple[BaseChatModel, str]:
    """
    get the specified LLM provider baseclass
    Args:
        model: string containing model name

    Returns:
    The Chat baseclass from langchain
    """
    provider_key = None
    # Extract the provider prefix from the model name
    for key in provider_mapping.keys():
        if model_mapping[key](model):
            provider_key = key
            break

    # Error handling if model does not exist
    if provider_key is None:
        raise ModelNotSupportedException(model)

    if not os.getenv(env_vars[provider_key]):
        raise ApiKeyMissingException(provider_key, env_vars[provider_key])

    return provider_mapping[provider_key](
        model=model, # type: ignore
        streaming=True,
    ), provider_key