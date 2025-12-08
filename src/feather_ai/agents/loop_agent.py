"""
A loop Agent that prompts the model again and again until either a specific condition is met or the maximum number of iterations is reached.
"""
import inspect
from typing import Callable, Optional, Tuple, Any, Dict, Coroutine
from .base_agent import BaseAgent


class LoopAgent(BaseAgent):
    def __init__(
            self,
            agent,
            stop_or_proceed: (
                    Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]
                    | Callable[[Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, Optional[str]]]]
            ),
            max_iterations: int,
    ):
        """
        init method for LoopAgent class
        Args:
            agent: ai agent with all functionalities you want
            stop_or_proceed: function that decides whether to stop the loop. True means stop, False means proceed. can be async if calling arun
                input:
                    dict with the following keys:
                        "prompt" with which the agent was called (str | Prompt)
                        "response" from the agent (AIResponse)
                        "iteration" number (int)
                returns a tuple of (should_stop: bool, new_prompt: str) or just True
        """
        self.agent = agent
        self.stop_or_proceed = stop_or_proceed
        self.is_async = inspect.iscoroutinefunction(stop_or_proceed)
        self.max_iterations = max_iterations

    def run(self, prompt):
        """
        Simple run method that loops the agent until the stopping condition is met.
        Args:
            prompt: string or prompt object that queries the agent

        Returns:
            AIResponse object
        """
        current_prompt = prompt
        current_response = None
        for i in range(self.max_iterations):
            # get current response from the agent
            current_response = self.agent.run(current_prompt)

            stop_response = self.stop_or_proceed({"prompt": current_prompt, "response": current_response, "iteration": i})
            # if stop_response returned only a bool
            if isinstance(stop_response, bool):
                if stop_response:
                    return current_response
                else:
                    continue

            should_stop, new_prompt = stop_response
            if should_stop:
                return current_response
            else:
                current_prompt = new_prompt

        return current_response

    async def arun(self, prompt):
        """
        Simple arun method that loops the agent until the stopping condition is met.
        Args:
            prompt: string or prompt object that queries the agent

        Returns:
            AIResponse object
        """
        current_prompt = prompt
        current_response = None
        for i in range(self.max_iterations):
            # get current response from the agent
            current_response = await self.agent.arun(current_prompt)

            stop_response = self.stop_or_proceed(
                {"prompt": current_prompt, "response": current_response, "iteration": i}) \
                if not self.is_async \
                else (await self.stop_or_proceed({"prompt": current_prompt, "response": current_response, "iteration": i}))
            # if stop_response returned only a bool
            if isinstance(stop_response, bool):
                if stop_response:
                    return current_response
                else:
                    continue

            should_stop, new_prompt = stop_response
            if should_stop:
                return current_response
            else:
                current_prompt = new_prompt

        return current_response

    async def stream(self, prompt, **kwargs):
        """
        Stream back the responses from the agent.
        No token by token streaming but instead a stream of the response every loop iteration.
        Args:
            prompt: string or prompt object that queries the agent

        Returns:
            AsyncGenerator with the following tuples:
            (iteration: int, "tool_call", ToolCall) if the model made a tool call in iteration i
            (iteration: int, "tool_response", ToolResponse) the response of the tool call that was made in iteration i
            (iteration: int, "response", AIResponse) the models response in iteration i
            (iteration: int, "stop_response", bool | Tuple[bool, str]) the return value of the stop_or_proceed function in iteration i
        """
        current_prompt = prompt
        current_response = None
        for i in range(self.max_iterations):
            # get current response from the agent
            async for response in self.agent.stream(current_prompt, "messages"):
                if response[0] == "response":
                    current_response = response[1]
                yield i, response[0], response[1]

            stop_response = self.stop_or_proceed(
                {"prompt": current_prompt, "response": current_response, "iteration": i}) \
                if not self.is_async \
                else (
                await self.stop_or_proceed({"prompt": current_prompt, "response": current_response, "iteration": i}))
            yield i, "stop_response", stop_response
            # if stop_response returned only a bool
            if isinstance(stop_response, bool):
                if stop_response:
                    return
                else:
                    continue

            should_stop, new_prompt = stop_response
            if should_stop:
                return
            else:
                current_prompt = new_prompt

        return