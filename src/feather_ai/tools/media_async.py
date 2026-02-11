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


def _create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context using certifi certificates."""
    return ssl.create_default_context(cafile=certifi.where())


async def _make_pexels_request(
    url: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Internal function to make authenticated requests to Pexels API.

    Args:
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

    ssl_context = _create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=REQUESTS_TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            return await response.json()


async def asearch_stock_images(
    query: str,
    per_page: int = 10,
    page: int = 1,
    orientation: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    locale: Optional[str] = None
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

    response = await _make_pexels_request(url, params)
    photos = response["photos"]

    def curate_str_photo(photo: Dict[str, Any]) -> str:
        return f"""
---
url: {photo["src"]["original"]}
alt: {photo["alt"][:-1]})
size: ({photo["width"]}, {photo["height"]})
---
"""

    photos_str = [curate_str_photo(photo) for photo in photos]
    return "".join(photos_str)


async def asearch_stock_videos(
    query: str,
    per_page: int = 7,
    page: int = 1,
    orientation: Optional[str] = None,
    size: Optional[str] = None,
    locale: Optional[str] = None
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

    response = await _make_pexels_request(url, params)

    def curate_str_video(video: Dict[str, Any], idx: int) -> str:
        return f"""
# Video {idx}:
Identifier: {video["url"].split("/")[-2]}
Available urls: {[v["link"] for v in video["video_files"]]}
"""

    video_strings = [curate_str_video(video, idx) for idx, video in enumerate(response["videos"])]
    return "".join(video_strings)


async def main():
    """Example usage demonstrating concurrent searches."""
    from dotenv import load_dotenv
    load_dotenv()

    # Simple usage
    result = await asearch_stock_videos("Halong Bay")
    print(result)

    # Run searches concurrently
    images_task = asearch_stock_images("mountain sunset")
    videos_task = asearch_stock_videos("ocean waves")

    images, videos = await asyncio.gather(images_task, videos_task)
    print("=== Images ===")
    print(images)
    print("=== Videos ===")
    print(videos)


if __name__ == "__main__":
    asyncio.run(main())