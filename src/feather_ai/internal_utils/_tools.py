"""
Helper functions for tool calling
"""
import asyncio
from typing import Callable, Any, get_type_hints, List, Tuple, Type, AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage
from langchain_core.tools import StructuredTool, BaseTool
from langchain_google_genai import Modality
from pydantic import create_model, BaseModel
import inspect
import logging

from ..types.response import ToolCall, ToolResponse
from ..types.response import EOS

logger = logging.getLogger(__name__)

from ._tracing import ToolTrace, get_tool_trace_from_langchain

def execute_tool(response, tools):
    """
    Helper function to execute tool calls in the response from the LLM.
    """
    messages = []
    # Following code was copied from the tutorial about MCP:
    # Check if the LLM made tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Calling the following tools: {response.tool_calls}")
        # Add the assistant message with tool calls to our conversation
        messages.append(response)

        # Execute each tool call and create proper tool messages
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            # Find and execute the tool
            tool_result = None
            for tool in tools:
                if hasattr(tool, 'coroutine') and tool.coroutine is not None:
                    raise ValueError("You cannot use the normal run method with asynchronous tools. Use the arun method instead.")
                if tool.name == tool_name:
                    try:
                        tool_result = tool.run(tool_args)
                    except Exception as tool_error:
                        tool_result = f"Error executing tool: {tool_error}"
                    break

            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"

            # Create a tool message
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
            messages.append(tool_message)

    return messages


async def async_execute_tool(response, tools):
    """
    Helper function to execute tool calls in the response from the LLM.
    Calls all tools asynchronously in parallel.
    """
    messages = []

    # Check if the LLM made tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Calling the following tools: {response.tool_calls}")
        # Add the assistant message with tool calls to our conversation
        messages.append(response)

        # Create async tasks for all tool calls
        async def execute_single_tool(tool_call):
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            # Find and execute the tool
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # Check if tool has a coroutine (async) or func (sync)
                        if hasattr(tool, 'coroutine') and tool.coroutine is not None:
                            tool_result = await tool.arun(tool_args)
                        else:
                            logger.warning("Using synchronous tools will lead to sequential tool execution. Consider using asynchronous tools instead.")
                            tool_result = tool.run(tool_args)
                    except Exception as tool_error:
                        tool_result = f"Error executing tool: {tool_error}"
                    break

            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"

            # Create and return a tool message
            return ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )

        # Execute all tools in parallel
        tool_messages = await asyncio.gather(
            *[execute_single_tool(tc) for tc in response.tool_calls]
        )

        messages.extend(tool_messages)

    return messages

async def async_react_agent_with_tooling(
        llm: BaseChatModel, tools: List[BaseTool],
        messages: List[BaseMessage],
        structured_output: bool = False,
) -> Tuple[AIMessage | Type[BaseModel], List[ToolTrace]]:
    """
    Agent that can call tools in multiple rounds.
    Args:
        llm: langchain chat model
        tools: tools to be called by the chat model
        messages: current conversation
        structured_output: optional flag to indicate if the agent should return structured output

    Returns:

    """
    tool_calls = []
    while True:
        response = await llm.ainvoke(messages)

        # sorry for this hack, would not be necessary if langchain supported tool calls with structured output
        if structured_output and hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'respond':
                    tool_args = tool_call['args']
                    for tool in tools:
                        if tool.name == 'respond':
                            return tool.run(tool_args), tool_calls

        tool_messages = await async_execute_tool(response, tools)
        tool_calls.extend(get_tool_trace_from_langchain(response, tool_messages))
        if not tool_messages:
            if response.content:
                return response, tool_calls
            else:
                response.content = messages[-1].content
                return response, tool_calls
        messages.extend(tool_messages)

def react_agent_with_tooling(
        llm: BaseChatModel,
        tools: List[BaseTool],
        messages: List[BaseMessage],
        structured_output: bool = False,
) -> Tuple[AIMessage | Type[BaseModel], List[ToolTrace]]:
    """
    Agent that can call tools in multiple rounds.
    Args:
        llm: langchain chat model
        tools: tools to be called by the chat model
        messages: current conversation
        structured_output: optional flag to indicate if the agent should return structured output

    Returns:

    """
    tool_calls = []
    while True:
        response = llm.invoke(messages)

        # sorry for this hack, would not be necessary if langchain supported tool calls with structured output
        if structured_output and hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'respond':
                    tool_args = tool_call['args']
                    for tool in tools:
                        if tool.name == 'respond':
                            return tool.run(tool_args), tool_calls

        tool_messages = execute_tool(response, tools)
        tool_calls.extend(get_tool_trace_from_langchain(response, tool_messages))
        if not tool_messages:
            if response.content:
                return response, tool_calls
            else:
                response.content = messages[-1].content
                return response, tool_calls
        messages.extend(tool_messages)


async def stream_react_agent_with_tooling(
        llm: BaseChatModel,
        tools: List[BaseTool],
        messages: List[BaseMessage],
        stream_mode: str = "tokens",
        structured_output: bool = False,
) -> AsyncGenerator[Tuple[str, str | ToolResponse | ToolCall], None]:
    """
    Complex function for streaming the responses from agents that can call tools in a meaningful way
    Args:
        llm: langchain chat model
        tools: list of tools to be called by the chat model
        messages: list of input messages
        stream_mode: "tokens" - returns tokens one by one, "messages" - streams messages e.g. tool calls but not tokenwise
        structured_output: not yet implemented, indicates if the agent should return structured output

    Returns:
        streams back the following tuples:
        ("tool_call", ToolCall) if the model made a tool call
        ("tool_response", ToolResponse) the response of the tool call that was made
        ("token", str) chunk of output text from the model if it did not make a tool call
        ("token", EOS) signals the end of a token stream by the LLM
        ("structured_response", BaseModel) if structured_output is True
    """
    while True:
        chunks = []
        has_tool_calls = False
        streamed_tokens = False

        async for chunk in llm.astream(messages):
            chunks.append(chunk)

            # Detect tool calls early
            if not has_tool_calls:
                if (hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks) or \
                        (hasattr(chunk, 'tool_calls') and chunk.tool_calls):
                    has_tool_calls = True

            # If no tool calls detected yet, stream tokens immediately
            if not has_tool_calls and chunk.content and stream_mode == "tokens":
                streamed_tokens = True
                yield "token", chunk.content[0].get('text', '') if isinstance(chunk.content, list) else chunk.content

        if streamed_tokens:
            yield "token", EOS

        # Reconstruct full message
        response = chunks[0]
        for chunk in chunks[1:]:
            response = response + chunk

        # If we detected tool calls, yield the complete message
        if has_tool_calls:
            # If structured output, yield the structured response from the respond tool and return
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'respond' and structured_output:
                    tool_args = tool_call['args']
                    for tool in tools:
                        if tool.name == 'respond':
                            yield "structured_response", tool.run(tool_args)
                            return
                yield "tool_call", ToolCall(**tool_call)

            # Execute tools and continue
            tool_messages = await async_execute_tool(response, tools)

            if not tool_messages:
                return

            for tool_msg in filter(lambda m: isinstance(m, ToolMessage), tool_messages):
                yield "tool_response", ToolResponse(content=tool_msg.content, tool_call_id=tool_msg.tool_call_id)

            messages.extend(tool_messages)
        else:
            if stream_mode == "messages":
                yield "response", response
                return
            # Already streamed tokens, just return if not image generator
            return


def make_tool(func: Callable) -> StructuredTool:
    """
    Convert any function into a LangChain StructuredTool.

    Args:
        func: Any callable function with type hints

    Returns:
        A LangChain StructuredTool wrapping the function
    """
    if isinstance(func, BaseTool):
        return func
    # Get function metadata
    tool_name = func.__name__
    tool_description = func.__doc__ or f"Run {func.__name__}"

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Build Pydantic model fields from parameters
    fields = {}
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, Any)
        param_default = ... if param.default == inspect.Parameter.empty else param.default
        fields[param_name] = (param_type, param_default)

    # Create Pydantic model for schema
    InputSchema = create_model(f"{tool_name}Input", **fields)

    # Use 'coroutine' parameter for async functions, 'func' for sync functions
    if asyncio.iscoroutinefunction(func):
        return StructuredTool(
            name=tool_name,
            description=tool_description,
            args_schema=InputSchema,
            coroutine=func
        )
    else:
        return StructuredTool(
            name=tool_name,
            description=tool_description,
            args_schema=InputSchema,
            func=func
        )