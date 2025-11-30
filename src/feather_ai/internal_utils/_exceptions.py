"""
Custom exceptions for feather_ai.
"""
from typing import Optional


class ModelNotSupportedException(Exception):
    """
    Raised when a user requests a model that the library does not support.

    Attributes:
        model_name (str): The name of the unsupported model.
    """

    def __init__(self, model_name: str, message: str | None = None):
        if message is None:
            message = f"Model '{model_name}' is not supported by feather_ai."
        super().__init__(message)
        self.model_name = model_name

class ApiKeyMissingException(Exception):
    """
    Raised when an API key for the requested provider is missing.
    """

    def __init__(self, provider: Optional[str] = None, env_key: Optional[str] = None, message: str | None = None):
        if message is None:
            message = f"API key for provider '{provider}' is missing. Please set the environment variable '{env_key}'."
        super().__init__(message)