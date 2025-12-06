import base64
import os

from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from google.genai import types
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-pro-image-preview")

message = {
    "role": "user",
    "content": "Generate a photorealistic image of a cuddly cat wearing a hat.",
}

# You need to pass response_modalities in generation_config
response = model.invoke(
    [message],
    response_modalities=[Modality.TEXT, Modality.IMAGE],
)

# The response.content is a list, access the first element directly
def _get_image_base64(response):
    # response.content is already a list, just get the first item
    for block in response.content:
        if isinstance(block, dict) and block.get("image_url"):
            return block["image_url"]["url"].split(",")[-1]
    raise ValueError("No image found in response")

image_base64 = _get_image_base64(response)
with open("cat_pro.png", "wb") as f:
    f.write(base64.b64decode(image_base64))