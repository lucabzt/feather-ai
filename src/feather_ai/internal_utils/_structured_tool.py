"""
This is a bit of a hacky way to combine tool calling with structured output.
Inspired by https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph
We use a respond() tool to return structured output.
"""

from typing import Type
from pydantic import BaseModel
from langchain_core.tools import StructuredTool


def get_respond_tool(schema: Type[BaseModel]) -> StructuredTool:
    """
    Generates a LangChain tool named 'respond' based on a Pydantic schema.
    """

    # 1. Define the function logic
    # We accept **kwargs so the function can accept any arguments defined in the schema.
    # We simply instantiate the Pydantic model with these arguments.
    def respond_func(**kwargs) -> BaseModel:
        return schema(**kwargs)

    # 2. Construct the dynamic docstring
    # Start with the fixed header required by the prompt
    docstring_parts = [
        "This is the tool you should use to respond to the user when you do not want to make any other tool calls. "
        "The user requested for structured output which will be provided through this tool."
        "Please always use this tool ALONE and only call it ONCE, without calling any other tools.",
        "",
        "Args:"
    ]

    # Iterate over the Pydantic model fields to generate the Args section
    # Note: Uses Pydantic v2 'model_fields'. For v1, use '__fields__'.
    field_names = []
    for name, field in schema.model_fields.items():
        field_names.append(name)
        description = field.description

        # If a description exists, append it; otherwise just list the name
        if description:
            docstring_parts.append(f"{name}: {description}")
        else:
            docstring_parts.append(f"{name}:")

    # Add the Returns section
    docstring_parts.append("")
    docstring_parts.append("Returns:")
    docstring_parts.append(f"A {schema.__name__} object with the fields {', '.join(field_names)}")

    # Join the parts to create the final docstring
    respond_func.__doc__ = "\n".join(docstring_parts)

    # 3. Create and return the StructuredTool
    # We pass the original schema as 'args_schema'. LangChain uses this to
    # generate the JSON Schema for the LLM (handling types, defaults, and optionals automatically).
    return StructuredTool.from_function(
        func=respond_func,
        name="respond",
        description=respond_func.__doc__,
        args_schema=schema
    )