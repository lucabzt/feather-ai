"""
Contains the Response class that summarizes a response from an AI Agent
"""
from typing import Optional, List, Type

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel
from ..internal_utils._tracing import ToolTrace


class UsageInfo(BaseModel):
    """
    Token usage information aggregated across one or more LLM invocations.

    Consumers can sum multiple UsageInfo objects (e.g. one per LLM call in a
    react/tool loop or one per streamed usage event) using ``+`` or ``merge``.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    calls: int = 0  # number of LLM invocations aggregated into this object

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        if other is None:
            return self
        if not isinstance(other, UsageInfo):
            return NotImplemented
        return UsageInfo(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            # keep whichever model name is known (prefer self)
            model=self.model or other.model,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other) -> "UsageInfo":
        # supports sum([...]) which starts from 0
        if other == 0 or other is None:
            return self
        return self.__add__(other)

    def merge(self, other: "UsageInfo") -> "UsageInfo":
        """Alias for ``self + other``."""
        return self.__add__(other)

    def __repr__(self):
        return (
            f"UsageInfo(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, "
            f"total_tokens={self.total_tokens}, cache_read_tokens={self.cache_read_tokens}, "
            f"model={self.model!r}, calls={self.calls})"
        )


class AIResponse:
    def __init__(self, content: str | BaseModel | bytes, tool_calls: Optional[List[ToolTrace]] = None, input_messages: Optional[List[BaseMessage]] = None, usage: Optional[UsageInfo] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.input_messages = input_messages
        self.usage = usage

    def __repr__(self):
        if self.tool_calls:
            return f"AIResponse(content={self.content}, tool_calls={[str(tool_call) for tool_call in self.tool_calls]})"
        else:
            return f"AIResponse(content={self.content})"

class ToolCall(BaseModel):
    name: str
    args: dict
    id: str = id

    def __repr__(self):
        return f"ToolCall(name={self.name}, args={self.args}, id={self.id})"

class ToolResponse(BaseModel):
    content: str
    tool_call_id: str

    def __repr__(self):
        return f"ToolResponse(content={self.content}, tool_call_id={self.tool_call_id})"

EOS = object() # used to signal the end of a token stream from an LLM
