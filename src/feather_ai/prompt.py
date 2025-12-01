from pathlib import Path
from typing import Union
from langchain_core.messages import HumanMessage
from .internal_utils._multimodal import get_content_claude, get_content_gemini
from .document import Document


class Prompt:
    """
    The prompt class can be used to create a multimodal prompt with documents, images, etc.
    """
    def __init__(
        self, 
        text: str, 
        documents: list[Union[str, Path, Document]] = None
    ):
        self.text = text
        self.documents: list[Document] = []
        
        if documents:
            for doc in documents:
                if isinstance(doc, (str, Path)):
                    self.documents.append(Document.from_path(doc))
                elif isinstance(doc, Document):
                    self.documents.append(doc)
                else:
                    raise TypeError(f"Unsupported document type: {type(doc)}")

            # create the messages (different for each provider)
            text_content = [
                {"type": "text", "text": text},
                {"type": "text", "text": "The user uploaded the following documents:"},
                {"type": "text", "text": f"[{[doc.filename for doc in self.documents]}]"}
            ]

            self.message_map = {
                "gemini": HumanMessage(
                    content=text_content + get_content_gemini(self.documents)
                ),
                "claude": HumanMessage(
                    content=text_content + get_content_claude(self.documents)
                )
            }
        else:
            self.message = HumanMessage(content=text)

    def get_message(self, provider: str) -> HumanMessage:
        if provider in ["mistral", "openai"] and hasattr(self, "message_map"):
            raise ValueError(
                f"Provider {provider} does not support multimodal prompts yet."
                f"Please process your PDF into text first and then give a simple text Prompt."
            )
        if self.documents:
            try:
                return self.message_map[provider]
            except KeyError:
                raise ValueError(f"Provider {provider} is not supported.")

        return self.message
