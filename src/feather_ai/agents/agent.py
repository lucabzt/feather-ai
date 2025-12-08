"""
This file contains the main Agent class provided by feather_ai.
The AIAgent class can be used to create intelligent agents that can perform various tasks.
Its main advantage over other agentic AI frameworks is its simplicity and ease of use.
"""
import base64
from abc import ABC
from typing import Optional, List, Callable, Any, Type, AsyncGenerator, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import Modality
from pydantic import BaseModel

from .base_agent import BaseAgent
from ..internal_utils._image_gen import _get_image_base64
from ..internal_utils._provider import get_provider
from ..types.response import AIResponse, ToolCall, ToolResponse
from ..internal_utils._structured_tool import get_respond_tool
from ..prompt import Prompt
from ..internal_utils._tools import make_tool, react_agent_with_tooling, \
    async_react_agent_with_tooling, stream_react_agent_with_tooling


class AIAgent(BaseAgent):
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
                 ):
        """
        Initializes a new Agent instance.

        Args:
            model (str): The LLM used by the agent.
        """
        self.model = model
        self.system_instructions = instructions
        self.structured_output = True if output_schema else False
        self.image_generator = "image" in model
        if self.structured_output and self.image_generator:
            raise ValueError("Structured output is not supported for image generation models.")
        if tools and self.image_generator:
            raise ValueError("Image generation models do not support tool calling.")
        provider_data = get_provider(model)
        self.llm: BaseChatModel | Runnable = provider_data[0]
        if self.structured_output and not tools:
            self.llm = self.llm.with_structured_output(output_schema)
        self.provider_str: str = provider_data[1]
        # Bind tools to LLM
        if tools:
            self.tools = [make_tool(tool) for tool in tools]
            if self.structured_output:
                self.tools.append(get_respond_tool(output_schema))
                self.llm: BaseChatModel = get_provider(model)[0].bind_tools(self.tools, tool_choice='any')  # type: ignore
            else:
                self.llm: BaseChatModel = get_provider(model)[0].bind_tools(self.tools) # type: ignore

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
            response, tool_calls = react_agent_with_tooling(self.llm, self.tools, messages, self.structured_output)
        elif self.image_generator:
            response = self.llm.invoke(
                messages,
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            )
        else:
            response = self.llm.invoke(messages)

        ## Check for structured output
        if self.structured_output:
            return AIResponse(response, tool_calls, messages)
        elif self.image_generator:
            return AIResponse(base64.b64decode(_get_image_base64(response.content)), tool_calls, messages)
        else:
            return AIResponse(response.content, tool_calls, messages)

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
            response, tool_calls = await async_react_agent_with_tooling(self.llm, self.tools, messages, self.structured_output)
        elif self.image_generator:
            response = await self.llm.ainvoke(
                messages,
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            )
        else:
            response = await self.llm.ainvoke(messages)

        ## Check for structured output
        if self.structured_output:
            return AIResponse(response, tool_calls, messages)
        elif self.image_generator:
            return AIResponse(base64.b64decode(_get_image_base64(response)), tool_calls, messages)
        else:
            return AIResponse(response.content, tool_calls, messages)

    async def stream(self, prompt: Prompt | str, stream_mode: Optional[str] = "tokens") -> AsyncGenerator[Tuple[str, str | ToolCall | ToolResponse | BaseModel | bytes | AIResponse], None]:
        """
        Token by token streaming of the response from the agent.
        Args:
            prompt (Prompt | str): The prompt to process.
            stream_mode: "tokens" - returns tokens one by one, "messages" - streams messages e.g. tool calls but not tokenwise
        Returns:
            streams back one of the following tuples:
            ("tool_call", ToolCall) if the model made a tool call
            ("tool_response", ToolResponse) the response of the tool call that was made
            ("token", str) chunk of output text from the model if it did not make a tool call
            ("token", EOS) signals the end of a token stream by the LLM
            ("structured_response", YourStructuredClass) if structured_output is True
            ("image", bytes) if the model is an image generator
            ("response", AIResponse) if stream_mode is "messages" we return the entire response
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_instructions if self.system_instructions else ""),
        ]

        ## Check if user passed a Prompt Object or a string
        if isinstance(prompt, Prompt):
            messages.append(prompt.get_message(self.provider_str))
        else:
            messages.append(HumanMessage(content=prompt))

        ## Call tools if any
        if hasattr(self, "tools"):
            async for chunk in stream_react_agent_with_tooling(self.llm, self.tools, messages, stream_mode=stream_mode, structured_output=self.structured_output):
                yield chunk
        else:
            if self.image_generator:
                response = await self.llm.ainvoke(messages, response_modalities=[Modality.TEXT, Modality.IMAGE])
                yield "image", base64.b64decode(_get_image_base64(response))
                return
            if stream_mode == "messages":
                yield "response", await self.arun(prompt)
                return
            async for token in self.llm.astream(messages):
                yield "token", token.content