"""
Contains the Response class that summarizes a response from an AI Agent
"""
from typing import Optional, List, Type

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel
from src.feather_ai.internal_utils._tracing import ToolTrace


class AIResponse:
    def __init__(self, content: str | BaseModel, tool_calls: Optional[List[ToolTrace]] = None, input_messages: Optional[List[BaseMessage]] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.input_messages = input_messages

    def __repr__(self):
        if self.tool_calls:
            return f"AIResponse(content={self.content}, tool_calls={[str(tool_call) for tool_call in self.tool_calls]})"
        else:
            return f"AIResponse(content={self.content})"

class ToolCall:
    def __init__(self, name: str, args: dict, id: str, **kwargs):
        self.name = name
        self.args = args
        self.id = id

    def __repr__(self):
        return f"ToolCall(name={self.name}, args={self.args}, id={self.id})"

class ToolResponse:
    def __init__(self, content: str, tool_call_id: str):
        self.content = content
        self.tool_call_id = tool_call_id

    def __repr__(self):
        return f"ToolResponse(content={self.content}, tool_call_id={self.tool_call_id})"

EOS = object() # used to signal the end of a token stream from an LLM
