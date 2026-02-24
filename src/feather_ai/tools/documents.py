"""
Exposes some tools to analyze pdf documents
"""
from typing import Annotated, List, Optional

from feather_ai import AIAgent
from langchain_core.tools import InjectedToolArg, tool
import fitz
from ..document import Document

@tool
async def get_images_in_pages(
    file_name: str,
    start_page: Optional[int],
    end_page: Optional[int],
    documents: Annotated[List[Document], InjectedToolArg],
) -> str:
    """
    Tool for retrieving images and inferred captions from pdf documents.

    Args:
        file_name: The name of the pdf document
        start_page: First page to analyze (1-based)
        end_page: Last page to analyze (1-based)

    Returns:
        A string listing all images found with inferred captions
    """

    relevant_document = next(
        (doc for doc in documents if doc.filename == file_name),
        None
    )

    if relevant_document is None:
        return "Error: Document not found. Please use the correct file_name."

    if relevant_document.mime_type != "application/pdf":
        return "Error: Document is not a PDF."

    doc = fitz.open(stream=relevant_document.content, filetype="pdf")

    start_page = start_page if start_page is not None else 1
    end_page = end_page if end_page is not None else doc.page_count

    image_id = 1
    results = []

    for page_number in range(start_page - 1, min(end_page, doc.page_count)):
        page = doc.load_page(page_number)

        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        image_blocks = [b for b in blocks if b[6] == 1]

        for img in image_blocks:
            img_rect = fitz.Rect(img[:4])

            caption = None
            min_distance = float("inf")

            for tb in text_blocks:
                tb_rect = fitz.Rect(tb[:4])

                # Prefer text directly BELOW the image
                if tb_rect.y0 >= img_rect.y1:
                    distance = tb_rect.y0 - img_rect.y1
                    if distance < min_distance:
                        min_distance = distance
                        caption = tb[4].strip()

            if caption:
                caption = caption.replace("\n", " ").strip()
            else:
                caption = "No caption found"

            results.append(f"ID={image_id}, Caption: {caption}")
            image_id += 1

    if not results:
        return "No images found in the specified page range."

    return "\n".join(results)


def _get_pdf_pages_content(
    file_name: str,
    start_page: Optional[int],
    end_page: Optional[int],
    documents: List[Document],
) -> str:
    """
    Helper function for retrieving content as text from pdf documents.
    Used by both get_pdf_pages and summarize_pdf_pages tools.
    """
    relevant_document = next(
        (doc for doc in documents if doc.filename == file_name),
        None
    )

    # Error Handling
    if relevant_document is None:
        return "Error: Document not found. Please use the correct file_name."

    if relevant_document.mime_type != "application/pdf":
        return "Error: Document is not a PDF."

    doc = fitz.open(stream=relevant_document.content, filetype="pdf")

    # Handle pages to be returned - ensure they're integers
    start_page = int(start_page) if start_page is not None else 1
    end_page = int(end_page) if end_page is not None else doc.page_count

    text = ""
    for page_number in range(start_page-1, min(end_page, doc.page_count)):
        page = doc.load_page(page_number)
        text += f"Page {page_number + 1}: {page.get_text()}\n"

    return text


@tool
async def get_pdf_pages(
    file_name: str,
    start_page: Optional[int],
    end_page: Optional[int],
    documents: Annotated[List[Document], InjectedToolArg],
) -> str:
    """
    Tool for retrieving content as text from pdf documents.
    Args:
        file_name: The name of the pdf document you want to retrieve content from.
        start_page: The first page you want to get content from.
        end_page: The last page you want to get content from.
    Returns:
        A string representing the content of the pdf document, page by page.
    """
    return _get_pdf_pages_content(file_name, start_page, end_page, documents)

@tool
async def get_document_chars(
    file_name: str,
    start_char: Optional[int],
    end_char: Optional[int],
    documents: Annotated[List[Document], InjectedToolArg],
) -> str:
    """
    Tool for retrieving character ranges from text-based documents.
    Args:
        file_name: The name of the document you want to retrieve content from.
        start_char: The starting character index (0-based, inclusive).
        end_char: The ending character index (0-based, exclusive).
    Returns:
        A string representing the selected character range of the document.
    """
    relevant_document = next(
        (doc for doc in documents if doc.filename == file_name),
        None
    )

    # Error Handling
    if relevant_document is None:
        return "Error: Document not found. Please use the correct file_name."

    allowed_types = {"text/plain", "application/json", "text/csv"}
    if relevant_document.mime_type not in allowed_types:
        return (
            "Error: Unsupported document type. "
            "Supported types are text/plain, application/json, text/csv."
        )

    try:
        content = relevant_document.content.decode("utf-8")
    except Exception:
        return "Error: Unable to decode document content as UTF-8 text."

    # Handle character bounds
    start_char = start_char if start_char is not None else 0
    end_char = end_char if end_char is not None else len(content)

    # Validate bounds
    if start_char < 0 or end_char < 0:
        return "Error: start_char and end_char must be non-negative."

    if start_char > end_char:
        return "Error: start_char cannot be greater than end_char."

    # Clamp end_char to document length
    end_char = min(end_char, len(content))

    return content[start_char:end_char]

@tool
async def summarize_pdf_pages(
    file_name: str,
    start_page: Optional[int],
    end_page: Optional[int],
    summary_agent: Annotated[AIAgent, InjectedToolArg],
    documents: Annotated[List[Document], InjectedToolArg],
) -> str:
    """
    Tool for summarizing pages from pdf documents usign a specialized agent.
    Can be used when there are too many pages to process in one call.

    Args:
        file_name: The name of the pdf document
        start_page: First page to analyze (1-indexed)
        end_page: Last page to analyze (1-indexed)

    Returns:
        A string giving a summary of the selected pages.
    """
    # Get the PDF pages content using the helper function
    pages = _get_pdf_pages_content(file_name, start_page, end_page, documents)
    summary = await summary_agent.arun(f"Summarize the following pages:\n\n{pages}")

    return summary