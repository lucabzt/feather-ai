"""
Provides a standardized Document class for multimodal prompts.
Can either be called by the user or automatically from the Prompt class.
"""
from dataclasses import dataclass
from io import BytesIO
import mimetypes
from pathlib import Path


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: bytes
    mime_type: str
    filename: str = ""

    @classmethod
    def from_path(cls, path: str | Path) -> "Document":
        """Create document from file path."""
        path = Path(path)
        content = path.read_bytes()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return cls(content=content, mime_type=mime_type, filename=path.name)

    @classmethod
    def from_bytes(cls, content: bytes, mime_type: str, filename: str | None = None) -> "Document":
        """Create document from bytes."""
        return cls(content=content, mime_type=mime_type, filename=filename)

    @classmethod
    def from_bytesio(cls, buffer: BytesIO, mime_type: str, filename: str | None = None) -> "Document":
        """Create document from BytesIO."""
        content = buffer.getvalue()
        return cls(content=content, mime_type=mime_type, filename=filename)