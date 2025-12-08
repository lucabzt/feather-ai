"""
Process pdfs and return chunks with rich metadata like page coordinates
"""
import os
from typing import List, Tuple

import pymupdf
from google.cloud import vision
from dotenv import load_dotenv
from dataclasses import dataclass
from src.feather_ai import Document
load_dotenv()
import logging
logging.basicConfig(level=logging.ERROR)

@dataclass
class TextChunk:
    text: str
    page: int
    bbox: List[float]

class DocumentChunker:
    def __init__(self, service_account_key: str = None):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
        self.google_vision_client = vision.ImageAnnotatorClient()

    def _ocr(self, page: int, image_bytes: bytes):
        """Detects text in the file."""
        image = vision.Image(content=image_bytes) # type: ignore

        response = self.google_vision_client.text_detection(image=image)
        text = response.text_annotations[0]
        bbox = text.bounding_poly.vertices[0]

        # return text objects with vertex.x, vertex.y = vertex in text.bounding_poly.vertices and text.description
        return TextChunk(
            text = text.description,
            page=page,
            bbox=[bbox.x, bbox.y],
        )

    def process_document_part(self, args: Tuple[str | Document, int, int]):
        document = args[0]
        start = args[1]
        end = args[2]
        if isinstance(document, str):
            document = Document.from_path(document)

        doc = pymupdf.Document(stream=document.content)

        for page_idx in range(start, end):
            ocr_result = self._ocr(page_idx, doc[page_idx].get_pixmap().tobytes())
            print(ocr_result)

        doc.close()

if __name__ == "__main__":
    account_key = "./service-account-key.json"
    chunker = DocumentChunker(account_key)
    chunker.process_document_part(("./ocr_pdf_test.pdf", 0, 12))







