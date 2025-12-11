"""
feather_ai.tools
================

Public API for the feather_ai tools package.
Provides ready-to-use tools for AI agents.
"""

from .web import (
    google_search,
    google_search_async,
    extract,
    extract_async,
    crawl,
    crawl_async,
    web_tools,
    web_tools_async,
)
from .code_execution import code_execution_python
from .media import search_stock_images, search_stock_videos
from .media_async import asearch_stock_images ,asearch_stock_videos

# Sync tools list
all_tools = [
    google_search,
    extract,
    crawl,
    code_execution_python,
]

# Async tools list
all_tools_async = [
    google_search_async,
    extract_async,
    crawl_async,
]

__all__ = [
    # Web tools (sync)
    "google_search",
    "extract",
    "crawl",
    # Web tools (async)
    "google_search_async",
    "extract_async",
    "crawl_async",
    # Web tool lists
    "web_tools",
    "web_tools_async",
    # media search
    "search_stock_images",
    "search_stock_videos",
    "asearch_stock_images",
    "asearch_stock_videos",
    # Code execution
    "code_execution_python",
    # All tools lists
    "all_tools",
    "all_tools_async",
]