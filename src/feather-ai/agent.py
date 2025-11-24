"""
This file contains the main Agent class provided by feather-ai.
The AIAgent class can be used to create intelligent agents that can perform various tasks.
Its main advantage over other agentic AI frameworks is its simplicity and ease of use.
"""

class AIAgent:
    """
    The AIAgent class represents an intelligent AI agent that can perform tool calling and give structured output.
    
    Attributes:
        model (str): The LLM used by the agent.
    """

    def __init__(self, model: str):
        """
        Initializes a new Agent instance.

        Args:
            model (str): The LLM used by the agent.
        """
        self.model = model