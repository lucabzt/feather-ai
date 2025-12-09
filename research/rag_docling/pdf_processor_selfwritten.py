"""
Process pdfs and return chunks with rich metadata like page coordinates
"""
import os
import time
from typing import List, Tuple

import pymupdf
from google.cloud import vision
from dotenv import load_dotenv
from dataclasses import dataclass

from pymupdf import Page

from src.feather_ai import Document
load_dotenv()

OCR_THRESHOLD = 50
CHUNK_SIZE = 1024

@dataclass
class TextChunk:
    text: str
    page_span: Tuple[int, int]
    bbox: List[dict[str, int]]

class DocumentChunker:
    def __init__(self, service_account_key: str = None):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
        self.google_vision_client = vision.ImageAnnotatorClient()

    def _ocr(self, page: int, image_bytes: bytes):
        """Detects text in the file."""
        image = vision.Image(content=image_bytes) # type: ignore

        response = self.google_vision_client.text_detection(image=image)
        text = response.text_annotations[0]

        vertices = [
            {"x": v.x, "y": v.y}
            for v in text.bounding_poly.vertices
        ]

        return TextChunk(
            text=text.description,
            page_span=(page, page),
            bbox=vertices
        )

    def _detect_ocr_needed(self, page: Page):
        # 1. Native Text Extraction
        raw_text = page.get_text()
        return len(raw_text) < OCR_THRESHOLD

    def process_document_part(self, args: Tuple[str | Document, int, int]):
        document = args[0]
        start = args[1]
        end = args[2]
        if isinstance(document, str):
            document = Document.from_path(document)

        doc = pymupdf.Document(stream=document.content)

        for page_idx in range(start, end):
            page = doc[page_idx]
            needs_ocr = self._detect_ocr_needed(page)
            if needs_ocr:
                ocr_result = self._ocr(page_idx, doc[page_idx].get_pixmap().tobytes())

        doc.close()

if __name__ == "__main__":
    account_key = "./service-account-key.json"
    chunker = DocumentChunker(account_key)
    chunker.process_document_part(("./docling.pdf", 0, 12))







