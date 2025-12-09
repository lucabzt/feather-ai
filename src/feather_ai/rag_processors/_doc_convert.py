import os
import markdown
from weasyprint import HTML
from ..document import Document

def _to_pdf(document: Document) -> Document:
    """
    Main entry point to turn any text-based document or image into PDF.
    """
    mime_type = document.mime_type

    # 1. Passthrough
    if mime_type == "application/pdf":
        return document

    # 3. Markdown
    elif mime_type in ["text/markdown", "text/x-markdown"]:
        text = document.content.decode("utf-8")
        html_content = markdown.markdown(text)
        pdf_bytes = HTML(string=html_content).write_pdf()
        return Document(
            content=pdf_bytes,
            mime_type="application/pdf",
            filename=os.path.splitext(document.filename)[0] + ".pdf",
        )

    # 4. Plain Text
    elif mime_type.startswith("text/"):
        text = document.content.decode("utf-8")
        html_content = f"<pre>{text}</pre>"
        pdf_bytes = HTML(string=html_content).write_pdf()
        return Document(
            content=pdf_bytes,
            mime_type="application/pdf",
            filename=os.path.splitext(document.filename)[0] + ".pdf",
        )

    else:
        raise ValueError(f"Unsupported MIME Type: {document.mime_type}")