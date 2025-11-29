"""
This file contains the main Agent class provided by feather_ai.
The AIAgent class can be used to create intelligent agents that can perform various tasks.
Its main advantage over other agentic AI frameworks is its simplicity and ease of use.
"""
from typing import Optional, List, Callable, Any, Dict, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.feather_ai.internal_utils._provider import get_provider
from src.feather_ai.internal_utils._response import AIResponse
from .prompt import Prompt
from .internal_utils._tools import make_tool, execute_tool, async_execute_tool
from .internal_utils._tracing import get_tool_trace_from_langchain


class AIAgent:
    """
    The AIAgent class represents an intelligent AI agent that can perform tool calling and give structured output.
    
    Attributes:
        model (str): The LLM used by the agent.
    """

    def __init__(self,
                 model: str,
                 instructions: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = None,
                 output_schema: Optional[Type[BaseModel]] = None,
                 tool_model: Optional[str] = None,
                 ):
        """
        Initializes a new Agent instance.

        Args:
            model (str): The LLM used by the agent.
        """
        self.model = model
        self.system_instructions = instructions
        self.structured_output = True if output_schema else False
        provider_data = get_provider(model)
        self.llm: BaseChatModel | Runnable = provider_data[0]
        if self.structured_output:
            self.llm = self.llm.with_structured_output(output_schema)
        self.provider_str: str = provider_data[1]
        # Bind tools to LLM
        if tools:
            self.tools = [make_tool(tool) for tool in tools]
            self.tool_llm: BaseChatModel = get_provider(tool_model)[0].bind_tools(self.tools) if tool_model else self.llm.bind_tools(self.tools) # type: ignore


    def run(self, prompt: Prompt | str):
        """
        Standard run method for the AIAgent class.
        Returns:
            AgentResponse object containing the agent's response.
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_instructions if self.system_instructions else ""),
        ]

        ## Check if user passed a Prompt Object or a string
        if isinstance(prompt, Prompt):
            messages.append(prompt.get_message(self.provider_str))
        else:
            messages.append(HumanMessage(content=prompt))

        tool_calls = None
        ## Call tools if any
        if hasattr(self, "tools"):
            tool_response = self.tool_llm.invoke(messages)
            tool_messages = execute_tool(tool_response, self.tools)
            messages.extend(tool_messages)
            tool_calls = get_tool_trace_from_langchain(tool_response, tool_messages)
            response = self.llm.invoke(messages)
        else:
            response = self.llm.invoke(messages)

        ## Check for structured output
        if self.structured_output:
            return AIResponse(response, tool_calls)
        else:
            return AIResponse(response.content, tool_calls)

    async def arun(self, prompt: Prompt | str):
        """
        Async run method for the AIAgent class.
        Returns:
            AgentResponse object containing the agent's response.
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_instructions if self.system_instructions else ""),
        ]

        ## Check if user passed a Prompt Object or a string
        if isinstance(prompt, Prompt):
            messages.append(prompt.get_message(self.provider_str))
        else:
            messages.append(HumanMessage(content=prompt))

        tool_calls = None
        ## Call tools if any
        if hasattr(self, "tools"):
            tool_response = await self.tool_llm.ainvoke(messages)
            tool_messages = await async_execute_tool(tool_response, self.tools)
            messages.extend(tool_messages)
            tool_calls = get_tool_trace_from_langchain(tool_response, tool_messages)
            response = await self.llm.ainvoke(messages)
        else:
            response = await self.llm.ainvoke(messages)

        ## Check for structured output
        if self.structured_output:
            return AIResponse(response, tool_calls)
        else:
            return AIResponse(response.content, tool_calls)