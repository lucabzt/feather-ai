"""
Base Agent class that agents inherit from
"""
from abc import abstractmethod, ABC
from typing import AsyncGenerator, Tuple
from pydantic import BaseModel

from ..prompt import Prompt
from ..types.response import ToolCall, ToolResponse


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.
    Defines the interface that all agent implementations must follow.
    """

    @abstractmethod
    def run(self, prompt: Prompt | str):
        """
        Standard run method for the agent.

        Args:
            prompt (Prompt | str): The prompt to process.

        Returns:
            AIResponse object containing the agent's response.
        """
        pass

    @abstractmethod
    async def arun(self, prompt: Prompt | str):
        """
        Async run method for the agent.

        Args:
            prompt (Prompt | str): The prompt to process.

        Returns:
            AIResponse object containing the agent's response.
        """
        pass

    @abstractmethod
    async def stream(self, prompt: Prompt | str, **kwargs):
        """
        Token by token streaming of the response from the agent.

        Args:
            prompt (Prompt | str): The prompt to process.

        Returns:
            AsyncGenerator yielding tuples of (type, content) where type can be:
            - "tool_call": followed by ToolCall
            - "tool_response": followed by ToolResponse
            - "token": followed by str chunk or "EOS"
            - "structured_response": followed by structured output
            - "image": followed by bytes
        """
        pass