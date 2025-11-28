"""
Custom exceptions for feather_ai.
"""

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