"""
Testing the AIAgent baseclass
Models:
claude-haiku-4-5
gpt-5-nano
gemini-2.5-flash-lite
mistral-small-2506
"""
import asyncio
import os
import time
from pprint import pprint

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging

from src.feather_ai.tools import search_stock_images
from src.feather_ai.tools.code_execution import code_execution_python
from src.feather_ai.tools.web import google_search, web_tools, web_tools_async
from src.feather_ai.types import EOS

logging.basicConfig(level=logging.ERROR)

from src.feather_ai import AIAgent
from src.feather_ai.prompt import Prompt
from src.feather_ai.utils import load_instruction_from_file

load_dotenv()

models = ["claude-haiku-4-5", "gpt-5-nano", "gemini-2.5-flash-lite", "mistral-small-2506"]
models_small = ["gemini-2.5-flash-lite"]
files = ["text_file.txt", "ocr_pdf_test.pdf", "ocr_image_test.jpeg"]

class Response(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(..., description="How confident from 0-1 you are in your answer")
def get_weather(location: str):
    return f"The weather in {location} is rainy today."
async def aget_weather(location: str):
    return f"The weather in {location} is rainy today."

def test_base_agent():
    question = "What is the capital of France?"
    print(question)
    for model in models:
        agent = AIAgent(model, instructions=load_instruction_from_file("test_instructions.txt"))
        resp = agent.run("What is the capital of France?")
        print(f"{model}: {resp.content}")

def test_multimodal():
    for model in models:
        print("-"*10 + f"Testing {model}" + "-"*10)
        prompt = Prompt(
            text="Please summarize the following documents each in 1 sentence",
            documents=[os.path.join("test_docs", file) for file in files]
        )
        agent = AIAgent(model)
        try:
            resp = agent.run(prompt)
        except ValueError as e:
            print(f"xxx Error for model {model}: {e}")
            continue
        print(resp.content)

def test_tool_calling():
    print("=== Testing Tool calling ===")
    for model in models_small:
        agent = AIAgent(model, tools=[search_stock_images])
        resp = agent.run("Give me only a single url of a bmw car")
        print(f"{model}: {resp.content}, Tool calls: {[str(tool_call) for tool_call in resp.tool_calls]}")
        pprint(resp.input_messages)

def test_structured_output():
    print("=== Testing Structured Output ===")
    for model in models:
        agent = AIAgent(model, output_schema=Response)
        resp = agent.run("What is the capital of France?")
        pprint(f"{model}: {resp.content}")
        assert isinstance(resp.content, Response)

def test_tool_calling_with_structured_output():
    print("=== Testing Tool calling with Structured Output ===")
    for model in models_small:
        agent = AIAgent(model, tools=[get_weather], output_schema=Response)
        resp = agent.run("What is the weather in Paris today?")
        pprint(f"{model}: {resp.content}")
        assert isinstance(resp.content, Response)

def test_multiple_tools():
    print("=== Testing Multiple tools ===")
    system_message = ("You are a helpful assistant that can interact with several tools."
                      "You will be given a question and you must answer it using the tools you have."
                      "First, just call the get_weather tool with the location as an argument."
                      "Then once you get an answer, use the get_tips tool to provide some tips based on the weather.")
    def get_tips(weather: str):
        return f"If its {weather}, bring an umbrella!"
    for model in models:
        agent = AIAgent(model, instructions=system_message, tools=[get_weather, get_tips], output_schema=Response)
        resp = agent.run("I am going to Paris today. What should I do?")
        pprint(f"{model}: {resp.content}")
        assert isinstance(resp.content, Response)

async def test_complex_async_tooling():
    print("=== Testing Multiple tools ===")
    system_message = ("You are a helpful assistant that can interact with several tools."
                      "You will be given a question and you must answer it using the tools you have."
                      "First, just call the get_weather tool with the location as an argument."
                      "Then once you get an answer, use the get_tips tool to provide some tips based on the weather.")

    async def get_weather(location: str):
        if location == "Munich":
            return f"The weather in Munich is sunny today."
        return f"The weather in {location} is rainy today."
    async def get_tips(weather: str, location: str):
        if weather == "sunny":
            return f"Go for a walk in the english garden!"
        return f"If its {weather} in {location}, bring an umbrella!"

    agent = AIAgent("gemini-2.5-flash-lite", instructions=system_message, tools=[get_weather, get_tips], output_schema=Response)
    resp = await agent.arun("Please check what my friends in Paris, Munich and Singapore should do today. Keep your answer in 3 short sentences.")
    print(resp)
    assert isinstance(resp.content, Response)

def test_async_run():
    async def test_run():
        agent = AIAgent("claude-haiku-4-5")
        resp = await agent.arun("What is the capital of France?")
        print(resp.content)

    asyncio.run(test_run())

def test_complex_tools():
    agent = AIAgent("gemini-2.5-flash-lite", tools=[*web_tools])
    resp = agent.run("Search for a python code execution tool in langchain. Use google_search and then extract from the provided urls.")
    print(resp)
    print("--------------------------------")
    pprint(resp.input_messages)

async def test_async_complex_tools():
    agent = AIAgent("gemini-2.5-flash-lite", tools=[code_execution_python, *web_tools_async])
    resp = await agent.arun(""
    "Please scrape this base url: https://docs.langchain.com/oss/python/langchain"
                            "and this: https://google.github.io/adk-docs/ for a code execution tool in both frameworks. Give me an overview which one is easier to use.")
    print(resp)
    print("--------------------------------")
    pprint(resp.input_messages)

async def test_streaming():
    agent = AIAgent("gpt-4o", tools=[*web_tools_async])
    chunks = []
    tool_calls = []
    max_length = 40
    async for chunk in agent.stream("whats the weather in kuala lumpur today?"):
        if chunk[0] == "tool_call":
            print(f"Model called tool: {chunk[1]}")
            tool_calls.append(chunk[1])
        if chunk[0] == "tool_response":
            response = chunk[1]
            tool_name = next((tc.name for tc in tool_calls if tc.id == response.tool_call_id), "unknown")
            print(f"Tool {tool_name} responded with: {repr(response)[:max_length] + ('...' if len(repr(response)) > max_length else '')}")
        if chunk[0] == "token":
            if chunk[1] is EOS:
                print("")
            else:
                print(chunk[1], end="", flush=True)
        if chunk[0] == "structured_response":
            print(chunk[1])
        chunks.append(chunk)

    """
    print("\n\n--------------------------------")
    print("SUMMARY (Token by Token):")
    # test if token for token streaming works (google does not support that)
    for chunk in chunks:
        print(chunk)
        time.sleep(0.2)
    """

def test_image_gen():
    agent = AIAgent("gemini-3-pro-image-preview")
    response = agent.run(Prompt(text="Please turn the images that I sent into an aerial drone image as if it was captured from above. The people in the second image should still just be recognizable in the drone image", documents=["krabi.jpg", "krabi_people.jpg"]))
    with open ("krabi_aerial.png", "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    test_image_gen()