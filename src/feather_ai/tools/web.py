import json
import os
from typing import Optional, List, Dict
from tavily import TavilyClient, AsyncTavilyClient

from ..internal_utils._exceptions import ApiKeyMissingException

_client: Optional[TavilyClient] = None
_async_client: Optional[TavilyClient] = None

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

    search_response = _client.search(query)

    # Curate the results into a markdown string
    def curate_results(result: dict):
        """Turn the search results into a markdown string for better readability"""
        return (f"Title: {result.get("title", "")}\nurl: {result.get("url", "")}\n"
                f"Content Snippets: {result.get("content", "")}")

    curated_results = [curate_results(result) for result in search_response["results"]]
    return "\n\n".join(curated_results)

async def google_search_async(query: str) -> str:
    """
        Simple google search tool for recent events and facts
        Performs a google search with the given query and returns a list of relevant search results
        Args:
            query: Your search query

        Returns:
            A curated list of relevant results
        """
    global _async_client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(
            message="I you want to use the google search tool, please set the environment variable TAVILY_API_KEY."
                    "You can get a free API key at https://www.tavily.com/")
    if not _async_client:
        _async_client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))

    search_response = await _async_client.search(query)

    # Curate the results into a markdown string
    def curate_results(result: Dict[str, str]):
        """Turn the search results into a markdown string for better readability"""
        return (f"Title: {result.get("title", "")}\nurl: {result.get("url", "")}\n"
                f"Content Snippets: {result.get("content", "")}")

    curated_results = [curate_results(result) for result in search_response["results"]]
    return "\n\n".join(curated_results)

def extract(urls: List[str]) -> List[Dict]:
    """
    Extracts raw content from the provided urls
    Args:
        urls: list of urls that should be extracted

    Returns:
        Dict with keys url, raw_content and favicon
    """
    global _client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(
            message="I you want to use web tools like extract, please set the environment variable TAVILY_API_KEY."
                    "You can get a free API key at https://www.tavily.com/")
    if not _client:
        _client = TavilyClient(os.getenv("TAVILY_API_KEY"))

    response = _client.extract(urls)

    # filter response to only include relevant keys
    keys = ["title", "url", "raw_content", "favicon"]
    errors = response.get("failed_results", [""])
    filtered_dicts = [{k: result.get(k) for k in keys if k in result} for result in response["results"]]

    return errors + filtered_dicts


async def extract_async(urls: List[str]) -> List[Dict]:
    """
    Extracts raw content from the provided urls
    Args:
        urls: list of urls that should be extracted

    Returns:
        Dict with keys url, raw_content and favicon
    """
    global _async_client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(
            message="I you want to use web tools like extract, please set the environment variable TAVILY_API_KEY."
                    "You can get a free API key at https://www.tavily.com/")
    if not _async_client:
        _async_client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))

    response = await _async_client.extract(urls)

    # filter response to only include relevant keys
    keys = ["title", "url", "raw_content", "favicon"]
    errors = response.get("failed_results", [""])
    filtered_dicts = [{k: result.get(k) for k in keys if k in result} for result in response["results"]]

    return errors + filtered_dicts

def crawl(base_url: str) -> List[str]:
    """
    Extracts all subpages of the given url
    Args:
        base_url: the base url to start the crawl from

    Returns:
        a list of suppages that can be extracted with the extract tool
    """
    global _client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(message="I you want to use the google search tool, please set the environment variable TAVILY_API_KEY."
                                             "You can get a free API key at https://www.tavily.com/")
    if not _client:
        _client = TavilyClient(os.getenv("TAVILY_API_KEY"))

    response = _client.map(base_url)

    return response["results"]


async def crawl_async(base_url: str) -> List[str]:
    """
    Extracts all subpages of the given url
    Args:
        base_url: the base url to start the crawl from

    Returns:
        a list of suppages that can be extracted with the extract tool
    """
    global _async_client
    if not os.getenv("TAVILY_API_KEY"):
        raise ApiKeyMissingException(message="I you want to use the google search tool, please set the environment variable TAVILY_API_KEY."
                                             "You can get a free API key at https://www.tavily.com/")
    if not _async_client:
        _async_client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))

    response = await _async_client.map(base_url)

    return response["results"]

web_tools = [google_search, extract, crawl]
web_tools_async = [google_search_async, extract_async, crawl_async]



if __name__ == "__main__":
    from dotenv import load_dotenv
    from pprint import pprint
    load_dotenv()
    result = google_search("Champions League Winner 2025")
    print(result)