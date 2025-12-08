"""
holds the code for the SubtaskExecutionAgent (see docstring)
"""
from typing import List, Any, Callable

from .base_agent import BaseAgent
from .. import AIResponse, Prompt


class SubtaskExecutionAgent(BaseAgent):
    """
    When working with multi-agent systems, there is often a typical orchestration pattern which emerges, that looks like this:
    Prompt -> Planner Agent -> Plan with multiple subtasks ->
    -> Subtask Agent 1
    -> Subtask Agent 2
    -> Subtask Agent N

    We simulate this pattern with a SubtaskExecutionAgent.
    """
    def __init__(
            self,
            planner_agent: BaseAgent,
            subtask_agents: BaseAgent | List[BaseAgent],
            task_creator: Callable[[AIResponse | Any], List[Prompt | str]]
    ):
        """
        init method for SubtaskExecutionAgent class
        Args:
            planner_agent: agent that plans the subtasks, usually returns structured output
            subtask_agents: agent(s) that execute the subtasks based on the planners output
            task_creator: function that creates the prompts for the subtasks from the response of the planner agent
        """
        self.planner_agent = planner_agent
        self.subtask_agents = subtask_agents
        self.task_creator = task_creator

    def run(self, prompt: str | Prompt) -> AIResponse:
        pass