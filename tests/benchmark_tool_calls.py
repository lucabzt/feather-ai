"""
Benchmark tool calling: sync vs async
Tests if parallel tool execution works correctly
"""
import asyncio
import time
from src.feather_ai import AIAgent

from dotenv import load_dotenv
load_dotenv()


# Define three mock tools that wait 3 seconds
async def mock_tool_1(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    await asyncio.sleep(3)
    return f"Tool 1 completed after 3 seconds with query: {query}"


async def mock_tool_2(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    await asyncio.sleep(3)
    return f"Tool 2 completed after 3 seconds with query: {query}"


async def mock_tool_3(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    await asyncio.sleep(3)
    return f"Tool 3 completed after 3 seconds with query: {query}"

# Define three mock tools that wait 3 seconds
def mock_tool_1_sync(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    time.sleep(3)
    return f"Tool 1 completed after 3 seconds with query: {query}"


def mock_tool_2_sync(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    time.sleep(3)
    return f"Tool 2 completed after 3 seconds with query: {query}"


def mock_tool_3_sync(query: str) -> str:
    """A mock tool that simulates a 3-second operation. You MUST call this tool."""
    time.sleep(3)
    return f"Tool 3 completed after 3 seconds with query: {query}"


async def benchmark_async():
    """Benchmark async tool calling (should take ~3 seconds with parallel execution)"""
    print("\n=== Testing ASYNC tool calling ===")

    agent = AIAgent(
        "gemini-2.5-flash-lite",
        tools=[mock_tool_1, mock_tool_2, mock_tool_3]
    )

    prompt = """Please call ALL THREE tools (mock_tool_1, mock_tool_2, and mock_tool_3) 
    with the query 'test'. You must call all three tools to complete this task 
    Then tell me what the three tools responded with."""

    start = time.time()
    resp = await agent.arun(prompt)
    elapsed = time.time() - start

    print(f"Async execution took: {elapsed:.2f} seconds")
    print(f"Response: {resp.content}\n")

    return elapsed


def benchmark_sync():
    """Benchmark sync tool calling (should take ~9 seconds with sequential execution)"""
    print("\n=== Testing SYNC tool calling ===")

    agent = AIAgent(
        "gemini-2.5-flash-lite",
        tools=[mock_tool_1_sync, mock_tool_2_sync, mock_tool_3_sync]
    )

    prompt = """Please call ALL THREE tools (mock_tool_1, mock_tool_2, and mock_tool_3) 
    with the query 'test'. You must call all three tools to complete this task.
    Then tell me what the three tools responded with."""

    start = time.time()
    resp = agent.run(prompt)
    elapsed = time.time() - start

    print(f"Sync execution took: {elapsed:.2f} seconds")
    print(f"Response: {resp.content}\n")

    return elapsed


async def main():
    print("=" * 60)
    print("TOOL CALLING BENCHMARK")
    print("Expected: Async ~3s (parallel), Sync ~9s (sequential)")
    print("=" * 60)

    # Run sync first
    sync_time = benchmark_sync()

    # Run async
    async_time = await benchmark_async()

    # Results
    print("=" * 60)
    print("RESULTS:")
    print(f"Sync execution: {sync_time:.2f} seconds")
    print(f"Async execution: {async_time:.2f} seconds")
    print(f"Speedup: {sync_time / async_time:.2f}x")
    print("=" * 60)

    if async_time < 5:
        print("✓ SUCCESS: Async execution is fast (parallel tool calls working!)")
    else:
        print("✗ FAILED: Async execution is too slow (tools may be running sequentially)")


if __name__ == "__main__":
    asyncio.run(main())