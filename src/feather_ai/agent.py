"""
This file contains the main Agent class provided by feather_ai.
The AIAgent class can be used to create intelligent agents that can perform various tasks.
Its main advantage over other agentic AI frameworks is its simplicity and ease of use.
"""
from typing import Optional, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from ._provider import get_provider
from ._response import AIResponse
from .prompt import Prompt


class AIAgent:
    """
    The AIAgent class represents an intelligent AI agent that can perform tool calling and give structured output.
    
    Attributes:
        model (str): The LLM used by the agent.
    """

    def __init__(self,
                 model: str,
                 system_instructions: Optional[str] = None,):
        """
        Initializes a new Agent instance.

        Args:
            model (str): The LLM used by the agent.
        """
        self.model = model
        self.system_instructions = system_instructions
        self.llm, self.provider_str = get_provider(model)

    def run(self, prompt: Prompt | str):
        """
        Standard run method for the AIAgent class.
        Returns:
            AgentResponse object containing the agent's response.
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_instructions if self.system_instructions else ""),
        ]
        if isinstance(prompt, Prompt):
            messages.append(prompt.get_message(self.provider_str))
        else:
            messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return AIResponse(response.content)