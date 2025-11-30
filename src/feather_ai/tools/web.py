import os
from typing import Optional
from tavily import TavilyClient

from src.feather_ai.internal_utils._exceptions import ApiKeyMissingException

_client: Optional[TavilyClient] = None

def google_search(query: str) -> str:
    """
    Simple google search tool for recent events and facts
    Args:
        query: Your search query

    Returns:
        A curated list of relevant results
    """
    global _client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(message="I you want to use the google search tool, please set the environment variable TAVILY_API_KEY."
                                             "You can get a free API key at https://www.tavily.com/")
    if not _client:
        _client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    return str(_client.search(query))