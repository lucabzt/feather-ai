"""
Testing the AIAgent baseclass
Models:
claude-haiku-4-5
gpt-5-nano
gemini-2.5-flash-lite
mistral-small-2506
"""
import os
from dotenv import load_dotenv

from src.feather_ai import AIAgent
from src.feather_ai.prompt import Prompt

load_dotenv()

models = ["claude-haiku-4-5", "gpt-5-nano", "gemini-2.5-flash-lite", "mistral-small-2506"]
files = ["text_file.txt", "ocr_pdf_test.pdf", "ocr_image_test.jpeg"]

def test_base_agent():
    question = "What is the capital of France?"
    print(question)
    for model in models:
        agent = AIAgent(model)
        resp = agent.run("What is the capital of France?")
        print(f"{model}: {resp.text}")

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
        print(resp.text)

if __name__ == "__main__":
    test_multimodal()