"""
Helper functions for tool calling
"""
import asyncio
from typing import Callable, Any, get_type_hints

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import create_model, Field
import inspect


def execute_tool(response, tools):
    """
    Helper function to execute tool calls in the response from the LLM.
    """
    messages = []
    # Following code was copied from the tutorial about MCP:
    # Check if the LLM made tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
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