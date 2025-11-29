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

from src.feather_ai import AIAgent
from src.feather_ai.prompt import Prompt
from src.feather_ai.utils import load_instruction_from_file

load_dotenv()

models = ["claude-haiku-4-5", "gpt-5-nano", "gemini-2.5-flash-lite", "mistral-small-2506"]
files = ["text_file.txt", "ocr_pdf_test.pdf", "ocr_image_test.jpeg"]

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
    def get_weather(location: str):
        return f"The weather in {location} is rainy today."
    for model in models:
        agent = AIAgent(model, tools=[get_weather])
        resp = agent.run("What is the weather in Paris right now? Keep your answer short.")
        print(f"{model}: {resp.content}, Tool calls: {[str(tool_call) for tool_call in resp.tool_calls]}")

def test_structured_output():
    class Response(BaseModel):
        answer: str = Field(..., description="The answer to the question")
        confidence: float = Field(..., description="How confident from 0-1 you are in your answer")
    print("=== Testing Structured Output ===")
    for model in models:
        agent = AIAgent(model, output_schema=Response)
        resp = agent.run("What is the capital of France?")
        pprint(f"{model}: {resp.content}")
        assert isinstance(resp.content, Response)

def test_async_run():
    async def test_run():
        agent = AIAgent("claude-haiku-4-5")
        resp = await agent.arun("What is the capital of France?")
        print(resp.content)

    asyncio.run(test_run())


if __name__ == "__main__":
    test_base_agent()