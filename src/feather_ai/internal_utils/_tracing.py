"""
Everything related to tracing LLM calls
"""
from dataclasses import dataclass
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage


def get_tool_trace_from_langchain(response: AIMessage, messages: list[BaseMessage]):
    tool_traces = []
    tool_calls = response.tool_calls
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]

    # error handling:
    if len(tool_calls) != len(tool_messages):
        raise ValueError("Number of tool calls and tool responses do not match."
                         f"len tool calls: {len(tool_calls)}, len responses: {len(tool_messages)}."
                         f"response: {response}, tool_messages: {tool_messages}")

    for idx, tool_call in enumerate(tool_calls):
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        tool_input = f"{tool_name}({tool_args})"
        output = tool_messages[idx].content
        tool_traces.append(ToolTrace(tool_name, tool_input, output))

    return tool_traces

@dataclass
class ToolTrace:
    tool_name: str
    input: str
    output: str

    def __str__(self) -> str:
        return f"{self.input}: {self.output}"