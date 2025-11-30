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
from pprint import pprint

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging

from src.feather_ai.tools.web import google_search

logging.basicConfig(level=logging.INFO)

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
    for model in models:
        agent = AIAgent(model, tools=[google_search])
        resp = agent.run("What is the weather in Paris today? Use google search to find the answer.")
        print(f"{model}: {resp.content}, Tool calls: {[str(tool_call) for tool_call in resp.tool_calls]}")

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


if __name__ == "__main__":
    test_tool_calling()