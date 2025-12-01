"""
Contains the Response class that summarizes a response from an AI Agent
"""
from typing import Optional, List, Type

from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from ._tracing import ToolTrace


class AIResponse:
    def __init__(self, content: str | BaseModel, tool_calls: Optional[List[ToolTrace]] = None, input_messages: Optional[List[BaseMessage]] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.input_messages = input_messages

    def __str__(self):
        if self.tool_calls:
            return f"AIResponse(content={self.content}, tool_calls={[str(tool_call) for tool_call in self.tool_calls]})"
        else:
            return f"AIResponse(content={self.content})"