"""
feather_ai
==========

Public API for the feather_ai package.
"""

from .agent import AIAgent          # Re-export class for top-level import
from ._exceptions import ModelNotSupportedException

__all__ = [
    "AIAgent",
    "ModelNotSupportedException",
]
