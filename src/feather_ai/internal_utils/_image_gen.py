"""
Helper functions for image generation
"""


def _get_image_base64(response):
    # response.content is already a list, just get the first item
    for block in response.content if hasattr(response, "content") else response:
        if isinstance(block, dict) and block.get("image_url"):
            return block["image_url"]["url"].split(",")[-1]
    raise ValueError("No image found in response")