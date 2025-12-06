<div align="center">
  <img src="design/feather-ai-logo.svg" alt="FeatherAI Logo" width="200"/>

  # FeatherAI

  **The lightest Agentic AI framework you'll ever see**

  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![GitHub](https://img.shields.io/badge/github-feather--ai-black.svg)](https://github.com/lucabzt/feather-ai)

</div>

---

## What is FeatherAI?

FeatherAI is a lightweight Python library designed to make building AI agents incredibly simple. Whether you're creating chatbots, automation tools, or complex multi-agent systems, FeatherAI provides an elegant API that gets out of your way.

### Key Features

- **Simple & Intuitive API** - Create powerful AI agents in just a few lines of code
- **Multi-Provider Support** - Works with OpenAI, Anthropic Claude, Google Gemini, and Mistral
- **Tool Calling** - Easily integrate custom functions and external APIs
- **Structured Output** - Get responses in validated Pydantic schemas
- **Multimodal Support** - Process text, images, and PDFs seamlessly
- **Async/Await Ready** - Built-in support for asynchronous execution
- **Built-in Tools** - Web search, code execution, and more out of the box
- **Lightweight** - Minimal dependencies, maximum performance

---

## Installation

```bash
pip install feather-ai-sdk
```

### Environment Setup

Create a `.env` file in your project root with your API keys:

```bash
# OpenAI (for GPT models)
OPENAI_API_KEY=your_openai_key_here

# Anthropic (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google (for Gemini models)
GOOGLE_API_KEY=your_google_key_here

# Mistral
MISTRAL_API_KEY=your_mistral_key_here

# For web search tools (optional)
TAVILY_API_KEY=your_tavily_key_here
```

> **Note:** You only need to set the API keys for the providers you'll use.

---

## Quick Start

### Basic Agent

```python
from feather_ai import AIAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create an agent
agent = AIAgent(model="gpt-4")

# Run the agent
response = agent.run("What is the capital of France?")
print(response.content)  # Output: Paris
```

### Agent with Instructions

```python
from feather_ai import AIAgent

# Create an agent with custom instructions
agent = AIAgent(
    model="claude-haiku-4-5",
    instructions="You are a helpful assistant that provides concise answers. Always explain concepts in simple terms."
)

response = agent.run("Explain quantum computing")
print(response.content)
```

### Agent with Tools

```python
from feather_ai import AIAgent

# Define a custom tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F"

# Create an agent with tools
agent = AIAgent(
    model="gpt-4",
    instructions="You are a helpful weather assistant. Use the available tools to answer questions.",
    tools=[get_weather]
)

response = agent.run("What's the weather like in San Francisco?")
print(response.content)
print(f"Tools called: {response.tool_calls}")
```

### Structured Output

```python
from feather_ai import AIAgent
from pydantic import BaseModel, Field

# Define your output schema
class WeatherResponse(BaseModel):
    location: str = Field(..., description="The location requested")
    temperature: int = Field(..., description="Temperature in Fahrenheit")
    conditions: str = Field(..., description="Weather conditions")
    confidence: float = Field(..., description="Confidence in answer (0-1)")

# Create agent with structured output
agent = AIAgent(
    model="gpt-4",
    output_schema=WeatherResponse
)

response = agent.run("What's the weather in Paris?")
print(response.content.location)      # Validated Pydantic object
print(response.content.temperature)   # Type-safe access
print(response.content.confidence)
```

### Multimodal Input

```python
from feather_ai import AIAgent, Prompt

# Create a prompt with documents
prompt = Prompt(
    text="Summarize these documents",
    documents=["report.pdf", "chart.png", "data.txt"]
)

agent = AIAgent(model="claude-sonnet-4-5")
response = agent.run(prompt)
print(response.content)
```

### Async Execution

```python
import asyncio
from feather_ai import AIAgent

async def main():
    agent = AIAgent(model="claude-haiku-4-5")
    response = await agent.arun("What is machine learning?")
    print(response.content)

asyncio.run(main())
```

---

## Supported Models

FeatherAI supports a wide range of LLM providers:

- **OpenAI:** `gpt-4`, `gpt-5-nano`, `gpt-4-turbo`, etc.
- **Anthropic:** `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4`, etc.
- **Google:** `gemini-2.5-flash-lite`, `gemini-pro`, etc.
- **Mistral:** `mistral-small-2506`, `mistral-large`, etc.

---

## Documentation

For detailed documentation, examples, and guides, visit our [documentation site](https://lucabzt.github.io/feather-ai/).

### Topics Covered:
- Getting Started
- System Instructions
- Tool Calling
- Structured Output
- Multimodal Input
- Native Tools
- Asynchronous Execution
- Real-World Examples

---

## Featured Projects

### 🍝 [Piatto Cooks](https://piatto-cooks.com/)
An AI-powered cooking assistant that helps you discover recipes, plan meals, and get personalized cooking guidance.
- Recipe Generation
- Meal Planning
- Dietary Preferences

### 🎓 [NexoraAI](https://www.nexora-ai.de/)
An intelligent mentoring platform that connects mentors and mentees, providing personalized guidance and learning paths.
- Personalized Learning
- Skill Assessment
- Progress Tracking

---

## Contributing

We welcome contributions! If you'd like to improve FeatherAI, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

FeatherAI is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- **GitHub:** [github.com/lucabzt/feather-ai](https://github.com/lucabzt/feather-ai)
- **Documentation:** [lucabzt.github.io/feather-ai](https://lucabzt.github.io/feather-ai/)

---

<div align="center">
  Made with ❤️ by the FeatherAI team
</div>
