"""
Pexels Stock Photo/Video Search Tools (Async Version)

A self-sufficient module for searching and retrieving stock photos and videos from Pexels.
These functions can be used directly as tools for agents without any MCP server dependencies.

Requirements:
    pip install aiohttp certifi

Environment Variables:
    PEXELS_API_KEY: Your Pexels API key (required)
    REQUESTS_TIMEOUT: Request timeout in seconds (default: 15)
"""

import os
import ssl
import asyncio
from typing import Dict, Any, Optional

import aiohttp
import certifi

REQUESTS_TIMEOUT = 15


async def _make_pexels_request(
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Internal function to make authenticated requests to Pexels API.

    Args:
        session: The aiohttp client session to use
        url: The Pexels API endpoint URL
        params: Query parameters for the request

    Returns:
        JSON response as a dictionary

    Raises:
        ValueError: If PEXELS_API_KEY is not set
        aiohttp.ClientError: If the request fails
    """
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_API_KEY:
        raise ValueError("PEXELS_API_KEY environment variable is not set")

    headers = {
        "Authorization": PEXELS_API_KEY,
    }

    async with session.get(url, headers=headers, params=params) as response:
        response.raise_for_status()
        return await response.json()


def _create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context using certifi certificates."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    return ssl_context


async def asearch_stock_images(
        query: str,
        per_page: int = 10,
        page: int = 1,
        orientation: Optional[str] = None,
        size: Optional[str] = None,
        color: Optional[str] = None,
        locale: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    Search for photos on Pexels.

    Args:
        query: Search query string
        per_page: Results per page (1-20, default: 10)
        page: Page number (minimum 1, default: 1)
        orientation: Image orientation - 'landscape', 'portrait', or 'square'
        size: Image size - 'large', 'medium', or 'small'
        color: Color filter - color name or hex code (e.g., 'red' or '#ff0000')
        locale: Locale code (e.g., 'en-US')
        session: Optional aiohttp session to reuse (creates one if not provided)

    Returns:
        String containing all search results with url, alt, width, height
    """
    url = "https://api.pexels.com/v1/search"
    params = {
        "query": query,
        "per_page": str(per_page),
        "page": str(page),
    }

    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size
    if color:
        params["color"] = color
    if locale:
        params["locale"] = locale

    def curate_str_photo(photo: Dict[str, Any], idx: int) -> str:
        return f"""
        # Photo {idx}:
        url: {photo["src"]["original"]}
        alt: {photo["alt"][:-1]})
        size: ({photo["width"]}, {photo["height"]})
        ---
        """

    if session is not None:
        response = await _make_pexels_request(session, url, params)
        photos = response["photos"]
        photos_str = [curate_str_photo(photo, idx) for idx, photo in enumerate(photos)]
        return "".join(photos_str)

    ssl_context = _create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=REQUESTS_TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as new_session:
        response = await _make_pexels_request(new_session, url, params)
        photos = response["photos"]
        photos_str = [curate_str_photo(photo, idx) for idx, photo in enumerate(photos)]
        return "".join(photos_str)


async def asearch_stock_videos(
        query: str,
        per_page: int = 7,
        page: int = 1,
        orientation: Optional[str] = None,
        size: Optional[str] = None,
        locale: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    Search for videos on Pexels.

    Args:
        query: Search query string
        per_page: Results per page (1-80, default: 7)
        page: Page number (minimum 1, default: 1)
        orientation: Video orientation - 'landscape', 'portrait', or 'square'
        size: Video size - 'large', 'medium', or 'small'
        locale: Locale code (e.g., 'en-US')
        session: Optional aiohttp session to reuse (creates one if not provided)

    Returns:
        All search results with identifier and urls
    """
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "per_page": str(per_page),
        "page": str(page),
    }

    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size
    if locale:
        params["locale"] = locale

    def curate_str_video(video: Dict[str, Any], idx: int) -> str:
        return f"""
        # Video {idx}:
        Identifier: {video["url"].split("/")[-2]}
        Available urls: {[v["link"] for v in video["video_files"]]}
        """

    if session is not None:
        response = await _make_pexels_request(session, url, params)
        video_strings = [curate_str_video(video, idx) for idx, video in enumerate(response["videos"])]
        return "".join(video_strings)

    ssl_context = _create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=REQUESTS_TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as new_session:
        response = await _make_pexels_request(new_session, url, params)
        video_strings = [curate_str_video(video, idx) for idx, video in enumerate(response["videos"])]
        return "".join(video_strings)


async def create_pexels_session() -> aiohttp.ClientSession:
    """
    Create a reusable aiohttp session configured for Pexels API.

    Use this when making multiple requests to avoid connection overhead.
    Remember to close the session when done.

    Returns:
        Configured aiohttp ClientSession

    Example:
        async with await create_pexels_session() as session:
            images = await search_stock_images("nature", session=session)
            videos = await search_stock_videos("ocean", session=session)
    """
    ssl_context = _create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=REQUESTS_TIMEOUT)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)