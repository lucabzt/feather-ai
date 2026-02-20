"""
First try to implement a streaming agent with structured output.
A current limitation of structured outputs is that it cannot be meaningfully streamed as
the json schema only makes sense after the full output is finished
We are trying to overcome this limitation by at least allowing Lists of structured output to be
streamed back.
Example:
    We have a schema
    class QuestionAnswerPair(BaseModel):
        question: str
        answer: str

    and want to generate multiple of these:
    List[QuestionAnswerPair]

Then you can now call
    agent = StructuredStreamingAgent(
        model="...",
        instructions="Generate a list of qa pairs for a testing the user on the given topic"
        output_schema=QuestionAnswerPair,
        tools="...",
    )
"""
import asyncio
from typing import Optional, List, Callable, Any, Type, AsyncGenerator, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from feather_ai.internal_utils._tools import make_tool, async_execute_tool
from feather_ai.prompt import Prompt
from feather_ai.internal_utils._provider import get_provider

import json
from typing import List, Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

from feather_ai.tools import web_tools_async
from feather_ai.types.response import ToolResponse, ToolCall

# Generic type variable to allow the function to return the specific model class passed in
T = TypeVar("T", bound=BaseModel)

# Exceptions that indicate transient network/server errors worth retrying
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

# extract from full parsed response
def extract(schema: Type[T], response: str) -> List[T]:
    """
    Extracts valid Pydantic objects from a partial JSON string containing a list.

    It specifically looks for a key "output_list", locates the starting bracket '[',
    and iterates through the string attempting to decode valid JSON objects one by one.
    Any incomplete JSON at the end is safely ignored.
    """

    # 1. Fail fast if the required key isn't present
    # Using a simple string find as per the prompt's allowed assumptions
    key_index = response.find('"output_list"')
    if key_index == -1:
        return []

    # 2. Locate the start of the list '['
    # We search starting from where we found the key
    start_bracket_index = response.find('[', key_index)
    if start_bracket_index == -1:
        return []

    # 3. Initialize the state for parsing
    # We start parsing immediately after the opening bracket
    current_index = start_bracket_index + 1
    decoder = json.JSONDecoder()
    valid_objects: List[T] = []

    length = len(response)

    while current_index < length:
        # 3a. Skip whitespace to find the next meaningful character
        while current_index < length and response[current_index].isspace():
            current_index += 1

        if current_index >= length:
            break

        # 3b. Check for list termination or separators
        char = response[current_index]

        if char == ']':
            # End of the list found
            break

        if char == ',':
            # Separator found, move to next character
            current_index += 1
            continue

        # 3c. Attempt to decode a single JSON object
        try:
            # raw_decode returns a tuple: (python_object, index_where_parsing_ended)
            obj, end_index = decoder.raw_decode(response, idx=current_index)

            # 3d. Validate against the provided Pydantic Schema
            try:
                pydantic_obj = schema.model_validate(obj)

                valid_objects.append(pydantic_obj)
            except ValidationError:
                # The JSON was valid, but it didn't match the schema.
                # We skip this object but continue parsing the list.
                pass

            # Update our index to where the decoder left off
            current_index = end_index

        except json.JSONDecodeError:
            break

    return valid_objects

# extract from chunk list and return only new outputs
def extract_structured_objects(
        schema: Type[BaseModel],
        extracted: int,
        chunks: List[str],
):
    full_str = "".join(chunks)
    extracted_objects = extract(schema, full_str)
    return extracted_objects[extracted:]

# Generate system instructions for the LLM
def generate_schema_instructions(model: Type[BaseModel]) -> str:
    schema = model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    instructions = f"""Your final response (excluding tool calls) must be valid JSON that conforms to the following JSON schema:
```json
{schema_str}
```

Important:
- Output ONLY the JSON object with no additional text before or after
- Ensure all required fields are present
- Use the correct data types as specified in the schema"""

    return instructions

async def stream_structured_output_with_tooling(
        llm: BaseChatModel,
        tools: List[BaseTool],
        messages: List[BaseMessage],
        schema: Type[BaseModel],
) -> AsyncGenerator[Tuple[str, str | ToolResponse | ToolCall | Any], None]:
    """
    Complex function for streaming the responses from agents that can call tools in a meaningful way
    Args:
        llm: langchain chat model
        tools: list of tools to be called by the chat model
        messages: list of input messages
        schema: the output schema the agent should adhere to

    Returns:
        streams back the following tuples:
        ("tool_call", ToolCall) if the model made a tool call
        ("tool_response", ToolResponse) the response of the tool call that was made
        ("structured_response", BaseModel) one chunk of output adhering to the provided schema
    """
    while True:
        chunks = []
        text_chunks = []
        extracted_objects = []
        has_tool_calls = False

        async for chunk in llm.astream(messages):
            print("THIS COMES FROM FEATHERAI! ", chunk)
            chunks.append(chunk)

            # Detect tool calls early
            if not has_tool_calls:
                if (hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks) or \
                        (hasattr(chunk, 'tool_calls') and chunk.tool_calls):
                    has_tool_calls = True

            # If no tool calls detected, stream structured_response immediately
            if not has_tool_calls and chunk.content:
                if isinstance(chunk.content, list):
                    text_chunks.extend(d["text"] for d in chunk.content if d["type"] == "text")
                else:
                    text_chunks.extend(chunk.content)
                objects = extract_structured_objects(schema, len(extracted_objects), text_chunks)
                extracted_objects.extend(objects)
                for obj in objects:
                    yield "structured_response", obj

        print("STREAM IS OVER FOR NOW")

        # Handle empty response - raise error to propagate to caller
        if not chunks:
            raise RuntimeError("LLM returned empty response (no chunks received)")

        # Reconstruct full message
        response = chunks[0]
        for chunk in chunks[1:]:
            response = response + chunk

        # If we detected tool calls, yield the complete message
        if has_tool_calls:
            # If structured output, yield the structured response from the respond tool and return
            for tool_call in response.tool_calls:
                yield "tool_call", ToolCall(**tool_call)

            # Execute tools and continue
            tool_messages = await async_execute_tool(response, tools)

            if not tool_messages:
                return

            for tool_msg in filter(lambda m: isinstance(m, ToolMessage), tool_messages):
                yield "tool_response", ToolResponse(content=tool_msg.content, tool_call_id=tool_msg.tool_call_id)

            messages.extend(tool_messages)
        else:
            return

# ACTUAL AGENT
class StructuredStreamingAgent:
    """
    First try to implement a streaming agent with structured output.
    Not very robust so use with caution and implement fail-safes.
    A current limitation of structured outputs is that it cannot be meaningfully streamed as
    the json schema only makes sense after the full output is finished
    We are trying to overcome this limitation by at least allowing Lists of structured output to be
    streamed back.
    Example:
        We have a schema
        class QuestionAnswerPair(BaseModel):
            question: str
            answer: str

        and want to generate multiple of these:
        List[QuestionAnswerPair]

    Then you can now call
        agent = StructuredStreamingAgent(
            model="...",
            instructions="Generate a list of qa pairs for a testing the user on the given topic"
            output_schema=QuestionAnswerPair,
            tools="...",
        )
    """
    def __init__(self,
                 model: str,
                 output_schema: Type[BaseModel],
                 instructions: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = None,
                 ):
        self.model = model
        self.system_instructions = instructions
        self.schema = output_schema

        class Output(BaseModel):
            output_list: List[output_schema]

        provider_data = get_provider(model)
        self.llm: BaseChatModel | Runnable = provider_data[0]
        self.provider_str: str = provider_data[1]
        if tools:
            self.tools = [make_tool(tool) for tool in tools]
            self.tool_llm: BaseChatModel | Runnable = self.llm.bind_tools(self.tools)
        self.schema_instructions = generate_schema_instructions(Output)

    async def stream(self, prompt: Prompt | str | List[BaseMessage], retries: int = 2) -> AsyncGenerator[Tuple[str, Any], None]:
        """
        Stream structured output from the agent.
        Args:
            prompt: The prompt to process.
            retries: Number of retries on network/server errors. Default is 0.
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_instructions if self.system_instructions else ""),
            SystemMessage(content=self.schema_instructions),
        ]

        ## Check if user passed a Prompt Object or a string
        if isinstance(prompt, Prompt):
            messages.append(prompt.get_message(self.provider_str))
        elif isinstance(prompt, str):
            messages.append(HumanMessage(content=prompt))
        else:
            messages.extend(prompt)

        for attempt in range(retries + 1):
            try:
                if hasattr(self, "tools"):
                    async for chunk in stream_structured_output_with_tooling(self.tool_llm, self.tools, messages, self.schema):
                        yield chunk
                else:
                    chunks = []
                    extracted_objects = []
                    async for chunk in self.llm.astream(messages):
                        chunks.append(chunk.content)
                        found_objects = extract_structured_objects(self.schema, len(extracted_objects), chunks)
                        extracted_objects.extend(found_objects)
                        for obj in found_objects:
                            yield "structured_response", obj
                return  # Success, exit retry loop
            except _RETRYABLE_EXCEPTIONS:
                if attempt == retries:
                    raise
                await asyncio.sleep(1)

async def main():
    # --- Setup Schema ---
    class QuestionAnswerPair(BaseModel):
        question: str
        answer: str

    agent = StructuredStreamingAgent(
        model="gemini-3-flash-preview",
        output_schema=QuestionAnswerPair,
        instructions="return min. 20 qa pairs about charlie kirk. ground in google search",
        tools=web_tools_async
    )

    async for chunk in agent.stream("rainbows"):
        print(chunk)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())