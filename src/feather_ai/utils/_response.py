"""
Contains the Response class that summarizes a response from an AI Agent
"""
from typing import Optional, List
from pydantic import BaseModel
from src.feather_ai.utils._tracing import ToolTrace


class AIResponse:
    def __init__(self, content: str | BaseModel, tool_calls: Optional[List[ToolTrace]] = None):
        self.content = content
        self.tool_calls = tool_calls

    def __str__(self):
        if self.tool_calls:
            return f"AIResponse(content={self.content}, tool_calls={self.tool_calls})"
        else:
            return f"AIResponse(content={self.content})"