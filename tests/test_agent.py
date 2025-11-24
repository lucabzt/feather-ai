"""
Testing the AIAgent baseclass
"""
from src.feather_ai import AIAgent
from dotenv import load_dotenv

load_dotenv()

def test_base_agent():
    agent = AIAgent("mistral-medium-2508")
    resp = agent.run("Give me an in-depth explanation of graph retrieval systems in AI Agents")
    print(resp.text)

if __name__ == "__main__":
    test_base_agent()