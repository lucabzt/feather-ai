#!/usr/bin/env python3
"""
Fast PDF Document Processor - RAG Optimized with Hybrid OCR
4-core CPU | 4GB RAM Target
Speed: < 20s per doc (Standard), Adaptive for Scanned Docs

Updated: Added meaningful chunk filtering for RAG systems
"""
import json
import time
import gc
import os
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import pymupdf
from dotenv import load_dotenv

from src.feather_ai import Document

# Load environment variables for Google Cloud
load_dotenv()
# Ensure credentials path is set if not already in env
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-key.json"

# --- Configuration ---
# Text Chunking Settings
CHUNK_TARGET_SIZE = 1024
CHUNK_OVERLAP = 100
MERGE_TOLERANCE = 40

# --- RAG Chunk Quality Settings ---
MIN_CHUNK_CHARS = 150  # Minimum characters for a chunk to be meaningful
MIN_CHUNK_WORDS = 20  # Minimum words for semantic meaning
MIN_ALPHA_RATIO = 0.5  # At least 50% of chars should be alphabetic
MAX_REPEAT_RATIO = 0.3  # Max 30% of text can be repeated patterns
MIN_UNIQUE_WORDS_RATIO = 0.3  # At least 30% of words should be unique
MIN_AVG_WORD_LENGTH = 2.5  # Filter out chunks with very short avg word length

# OCR Settings
OCR_TRIGGER_MIN_CHARS = 100
OCR_ZOOM_MATRIX = 2.0

# Image Filtering Settings
MIN_IMG_WIDTH = 150
MIN_IMG_HEIGHT = 150
MIN_IMG_BYTES = 2048
MAX_IMG_RATIO = 3.5
IMG_OUTPUT_DIR = "extracted_images"


@dataclass
class TextBlock:
    text: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    char_count: int
    word_count: int
    source: str = "native"


@dataclass
class ImageData:
    filename: str
    image_bytes: bytes
    format: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    width: int
    height: int
    size_bytes: int


@dataclass
class PageResult:
    page_num: int
    text_blocks: List[TextBlock]
    images: List[ImageData]
    processing_time: float
    ocr_performed: bool = False
    chunks_filtered: int = 0  # Track how many were dropped


def is_meaningful_chunk(text: str) -> Tuple[bool, str]:
    """
    Validates whether a text chunk is meaningful for RAG/vector search.
    Returns (is_valid, reason) tuple.
    """
    text = text.strip()

    # Basic length checks
    if len(text) < MIN_CHUNK_CHARS:
        return False, f"too_short_chars:{len(text)}"

    # Word count check
    words = text.split()
    word_count = len(words)
    if word_count < MIN_CHUNK_WORDS:
        return False, f"too_few_words:{word_count}"

    # Average word length (filters out gibberish like "a b c d e f g")
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    if avg_word_len < MIN_AVG_WORD_LENGTH:
        return False, f"avg_word_too_short:{avg_word_len:.1f}"

    # Alphabetic ratio (filters out chunks that are mostly numbers/symbols)
    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / len(text) if len(text) > 0 else 0
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False, f"low_alpha_ratio:{alpha_ratio:.2f}"

    # Unique words ratio (filters out repetitive content)
    unique_words = set(w.lower() for w in words if len(w) > 2)
    unique_ratio = len(unique_words) / word_count if word_count > 0 else 0
    if unique_ratio < MIN_UNIQUE_WORDS_RATIO:
        return False, f"too_repetitive:{unique_ratio:.2f}"

    # Check for repeated character patterns (e.g., "......" or "-----")
    repeat_pattern = re.findall(r'(.)\1{3,}', text)
    repeat_chars = sum(len(match) for match in repeat_pattern)
    if repeat_chars / len(text) > MAX_REPEAT_RATIO:
        return False, "repeated_chars"

    # Check if mostly whitespace or line breaks
    content_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    if content_chars / len(text) < 0.5:
        return False, "mostly_whitespace"

    # Passed all checks
    return True, "valid"


def clean_text_for_rag(text: str) -> str:
    """
    Cleans text to improve quality for embedding and retrieval.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive punctuation sequences
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', ' ', text)
    text = re.sub(r'[_]{3,}', ' ', text)

    # Remove standalone numbers that aren't part of meaningful content
    # (keeps numbers within sentences)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

    # Clean up result
    text = text.strip()

    return text


class FastPDFProcessor:
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or min(cpu_count(), 4)
        pymupdf.TOOLS.store_shrink(75)
        Path(IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def process_document(self, document: str | Document) -> Dict[str, Any]:
        start_time = time.time()
        if isinstance(document, str):
            document = Document.from_path(document)

        doc = pymupdf.Document(stream=document.content)
        total_pages = len(doc)
        metadata = doc.metadata
        doc.close()

        chunk_size = 50
        all_results = []

        print(f"Processing {total_pages} pages (Workers: {self.num_workers})...")

        for i in range(0, total_pages, chunk_size):
            chunk_args = [
                (document, p)
                for p in range(i, min(i + chunk_size, total_pages))
            ]

            with Pool(self.num_workers) as pool:
                chunk_results = pool.map(self._process_page_worker, chunk_args)

            all_results.extend(chunk_results)
            gc.collect()

        self._save_images(all_results)

        total_time = time.time() - start_time

        # Aggregation stats
        total_blocks = sum(len(p.text_blocks) for p in all_results)
        total_images = sum(len(p.images) for p in all_results)
        ocr_pages = sum(1 for p in all_results if p.ocr_performed)
        total_filtered = sum(p.chunks_filtered for p in all_results)

        print(f"✅ Done in {total_time:.2f}s")
        print(f"   📝 Meaningful Chunks: {total_blocks}")
        print(f"   🗑️  Filtered (too small/noisy): {total_filtered}")
        print(f"   👁️  OCR Triggered: {ocr_pages} pages")
        print(f"   🖼️  Saved Images: {total_images}")

        return {
            "metadata": metadata,
            "pages": all_results,
            "stats": {
                "total_pages": total_pages,
                "total_time": total_time,
                "total_images": total_images,
                "total_chunks": total_blocks,
                "filtered_chunks": total_filtered,
                "ocr_pages_count": ocr_pages
            }
        }

    def _save_images(self, results: List[PageResult]):
        for page in results:
            for img in page.images:
                out_path = Path(IMG_OUTPUT_DIR) / img.filename
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        f.write(img.image_bytes)

    @staticmethod
    def _perform_google_ocr(image_bytes: bytes) -> List[Dict]:
        """
        Calls Google Cloud Vision API.
        Returns a list of dicts with 'text' and 'vertices' (normalized 0-1 or absolute).
        """
        try:
            from google.cloud import vision
        except ImportError:
            print("❌ Google Cloud Vision library not found. Skipping OCR.")
            return []

        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=image_bytes)

            response = client.document_text_detection(image=image)

            if response.error.message:
                print(f"❌ Google OCR Error: {response.error.message}")
                return []

            return response.full_text_annotation.pages

        except Exception as e:
            print(f"❌ OCR Pipeline Exception: {e}")
            return []

    @staticmethod
    def _create_validated_chunk(
            text: str,
            bbox: Tuple[float, float, float, float],
            page_num: int,
            source: str
    ) -> Optional[TextBlock]:
        """
        Creates a TextBlock only if the text passes RAG quality checks.
        Returns None if the chunk should be filtered out.
        """
        # Clean the text first
        cleaned_text = clean_text_for_rag(text)

        # Validate
        is_valid, reason = is_meaningful_chunk(cleaned_text)

        if not is_valid:
            return None

        words = cleaned_text.split()
        return TextBlock(
            text=cleaned_text,
            bbox=bbox,
            page_num=page_num,
            char_count=len(cleaned_text),
            word_count=len(words),
            source=source
        )

    @staticmethod
    def _process_page_worker(args: Tuple) -> PageResult:
        document, page_num = args
        t_start = time.time()

        doc = pymupdf.Document(stream=document.content)
        page = doc[page_num]

        # 1. Native Text Extraction
        raw_blocks = page.get_text("blocks", sort=True)
        native_text_len = sum(len(b[4].strip()) for b in raw_blocks if b[6] == 0)

        chunks = []
        images = []
        ocr_triggered = False
        filtered_count = 0

        # --- OCR TRIGGER CHECK ---
        if native_text_len < OCR_TRIGGER_MIN_CHARS:
            ocr_triggered = True
            mat = pymupdf.Matrix(OCR_ZOOM_MATRIX, OCR_ZOOM_MATRIX)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height

            ocr_pages = FastPDFProcessor._perform_google_ocr(img_bytes)

            if ocr_pages:
                for ocr_page in ocr_pages:
                    for block in ocr_page.blocks:
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = "".join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "

                        block_text = block_text.strip()
                        if not block_text:
                            continue

                        xs = [v.x for v in block.bounding_box.vertices]
                        ys = [v.y for v in block.bounding_box.vertices]

                        if not xs or not ys:
                            continue

                        x0, x1 = min(xs) / scale_x, max(xs) / scale_x
                        y0, y1 = min(ys) / scale_y, max(ys) / scale_y

                        # Validate and create chunk
                        chunk = FastPDFProcessor._create_validated_chunk(
                            text=block_text,
                            bbox=(x0, y0, x1, y1),
                            page_num=page_num,
                            source="ocr"
                        )

                        if chunk:
                            chunks.append(chunk)
                        else:
                            filtered_count += 1
        else:
            # Standard Processing logic
            current_text = []
            current_bbox = list(raw_blocks[0][:4]) if raw_blocks else [0, 0, 0, 0]

            for block in raw_blocks:
                if block[6] != 0:
                    continue
                text = block[4].strip()
                if not text:
                    continue

                b_x0, b_y0, b_x1, b_y1 = block[:4]
                vertical_gap = b_y0 - current_bbox[3]

                is_new_section = vertical_gap > MERGE_TOLERANCE and len(current_text) > 0
                current_len = sum(len(t) for t in current_text)
                is_chunk_full = current_len > CHUNK_TARGET_SIZE

                if is_new_section or is_chunk_full:
                    full_text = " ".join(current_text)

                    # Validate and create chunk
                    chunk = FastPDFProcessor._create_validated_chunk(
                        text=full_text,
                        bbox=tuple(current_bbox),
                        page_num=page_num,
                        source="native"
                    )

                    if chunk:
                        chunks.append(chunk)
                    else:
                        filtered_count += 1

                    current_text = [text]
                    current_bbox = [b_x0, b_y0, b_x1, b_y1]
                else:
                    current_text.append(text)
                    current_bbox[0] = min(current_bbox[0], b_x0)
                    current_bbox[1] = min(current_bbox[1], b_y0)
                    current_bbox[2] = max(current_bbox[2], b_x1)
                    current_bbox[3] = max(current_bbox[3], b_y1)

            # Don't forget the last accumulated chunk
            if current_text:
                full_text = " ".join(current_text)
                chunk = FastPDFProcessor._create_validated_chunk(
                    text=full_text,
                    bbox=tuple(current_bbox),
                    page_num=page_num,
                    source="native"
                )

                if chunk:
                    chunks.append(chunk)
                else:
                    filtered_count += 1

        # --- Image Extraction (Standard) ---
        if not ocr_triggered:
            image_list = page.get_images(full=True)
            img_counter = 0
            for img_info in image_list:
                try:
                    xref = img_info[0]
                    w, h = img_info[2], img_info[3]
                    if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
                        continue

                    img_dict = doc.extract_image(xref)
                    img_bytes = img_dict["image"]
                    if len(img_bytes) < MIN_IMG_BYTES:
                        continue

                    rects = page.get_image_rects(xref)
                    bbox = tuple(rects[0]) if rects else (0.0, 0.0, 0.0, 0.0)

                    filename = f"p{page_num + 1:03d}_img{img_counter:02d}.{img_dict['ext']}"

                    images.append(ImageData(
                        filename=filename,
                        image_bytes=img_bytes,
                        format=img_dict['ext'],
                        bbox=bbox,
                        page_num=page_num,
                        width=w,
                        height=h,
                        size_bytes=len(img_bytes)
                    ))
                    img_counter += 1
                except Exception:
                    continue

        doc.close()

        return PageResult(
            page_num=page_num,
            text_blocks=chunks,
            images=images,
            processing_time=time.time() - t_start,
            ocr_performed=ocr_triggered,
            chunks_filtered=filtered_count
        )

    def export_json(self, result: Dict, filename: str = "output.json"):
        export_data = {
            "stats": result["stats"],
            "pages": []
        }
        for page in result["pages"]:
            page_dict = {
                "page": page.page_num,
                "ocr_performed": page.ocr_performed,
                "chunks_filtered": page.chunks_filtered,
                "text": [asdict(t) for t in page.text_blocks],
                "images": [
                    {k: v for k, v in asdict(img).items() if k != "image_bytes"}
                    for img in page.images
                ]
            }
            export_data["pages"].append(page_dict)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON metadata saved to {filename}")


if __name__ == "__main__":
    processor = FastPDFProcessor()
    result = processor.process_document("./bafög_antwort.docx")
    processor.export_json(result)