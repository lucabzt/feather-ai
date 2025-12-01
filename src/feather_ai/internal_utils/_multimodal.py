"""
All providers handle multimodality differently so this defines all the message types
"""
import base64
from typing import Dict, Any, List

from ..document import Document


def get_content_claude(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    helper to build multimodal content for Claude
    """
    content_claude = []
    for doc in documents:
        if doc.mime_type == "application/pdf":
            # PDFs use the document type
            content_claude.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(doc.content).decode("utf-8")
                }
            })
        elif doc.mime_type.startswith("image/"):
            # Images use the image type
            content_claude.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": doc.mime_type,
                    "data": base64.b64encode(doc.content).decode("utf-8")
                }
            })
        elif doc.mime_type.startswith("text/") or doc.mime_type in ["application/json", "application/xml"]:
            # Text files: decode and include as text
            try:
                text_content_decoded = doc.content.decode("utf-8")
                content_claude.append({
                    "type": "text",
                    "text": f"\n--- Content of {doc.filename or 'file'} ---\n{text_content_decoded}\n--- End of {doc.filename or 'file'} ---\n"
                })
            except UnicodeDecodeError:
                pass
        else:
            # Unsupported format for Claude
            raise ValueError(f"Unsupported file type for Claude: {doc.mime_type}")

    return content_claude

def get_content_gemini(documents: List[Document]) -> List[Dict[str, Any]]:
    """
        helper to build multimodal content for Gemini
    """
    content_gemini = [{
        "type": "media",
        "mime_type": doc.mime_type,
        "data": base64.b64encode(doc.content).decode("utf-8")
    } for doc in documents]

    return content_gemini