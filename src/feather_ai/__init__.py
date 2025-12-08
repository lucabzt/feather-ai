"""
feather_ai
==========

Public API for the feather_ai package.
"""

from .agents import AIAgent
from .document import Document
from .prompt import Prompt
from .utils import load_instruction_from_file
from .internal_utils._exceptions import ModelNotSupportedException
from .types.response import AIResponse

__all__ = [
    "AIAgent",
    "Document",
    "Prompt",
    "load_instruction_from_file",
    "ModelNotSupportedException",
    "AIResponse",
]
