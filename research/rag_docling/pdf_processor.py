#!/usr/bin/env python3
"""
Fast PDF Document Processor - RAG Optimized with Hybrid OCR
4-core CPU | 4GB RAM Target
Speed: < 20s per doc (Standard), Adaptive for Scanned Docs
"""
import json
import time
import gc
import os
import io
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import pymupdf
from dotenv import load_dotenv

# Load environment variables for Google Cloud
load_dotenv()
# Ensure credentials path is set if not already in env
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-key.json"

# --- Configuration ---
# Text Chunking Settings
CHUNK_TARGET_SIZE = 1000
CHUNK_OVERLAP = 100
MERGE_TOLERANCE = 15
MIN_CHUNK_CHARS = 50

# OCR Settings
OCR_TRIGGER_MIN_CHARS = 100  # <--- NEW: Trigger OCR if page has less text than this
OCR_ZOOM_MATRIX = 2.0  # <--- NEW: Upscale factor for better OCR accuracy (2.0 = 200 DPI approx)

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
    source: str = "native"  # <--- NEW: Track if text came from 'native' or 'ocr'


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


class FastPDFProcessor:
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or min(cpu_count(), 4)
        pymupdf.TOOLS.store_shrink(75)
        Path(IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        pdf_path = str(Path(pdf_path).resolve())

        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        metadata = doc.metadata
        doc.close()

        chunk_size = 50
        all_results = []

        print(f"Processing {total_pages} pages (Workers: {self.num_workers})...")

        for i in range(0, total_pages, chunk_size):
            chunk_args = [
                (pdf_path, p)
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

        print(f"✅ Done in {total_time:.2f}s")
        print(f"   📝 Text Chunks: {total_blocks}")
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
                "ocr_pages_count": ocr_pages
            }
        }

    def _save_images(self, results: List[PageResult]):
        for page in results:
            for img in page.images:
                out_path = Path(IMG_OUTPUT_DIR) / img.filename
                # Write only if it doesn't exist to save IO
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

            # Using document_text_detection for better density handling
            response = client.document_text_detection(image=image)

            if response.error.message:
                print(f"❌ Google OCR Error: {response.error.message}")
                return []

            # We extract full blocks to map to our chunking strategy
            # Text annotations[0] is the full text, but we want blocks for layout
            return response.full_text_annotation.pages

        except Exception as e:
            print(f"❌ OCR Pipeline Exception: {e}")
            return []

    @staticmethod
    def _process_page_worker(args: Tuple) -> PageResult:
        pdf_path, page_num = args
        t_start = time.time()

        doc = pymupdf.open(pdf_path)
        page = doc[page_num]

        # 1. Native Text Extraction
        raw_blocks = page.get_text("blocks", sort=True)
        native_text_len = sum(len(b[4].strip()) for b in raw_blocks if b[6] == 0)

        chunks = []
        images = []
        ocr_triggered = False

        # --- OCR TRIGGER CHECK ---
        if native_text_len < OCR_TRIGGER_MIN_CHARS:
            ocr_triggered = True
            # Render page to image for OCR
            # Matrix(2,2) doubles resolution for better OCR accuracy
            mat = pymupdf.Matrix(OCR_ZOOM_MATRIX, OCR_ZOOM_MATRIX)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # Helper to map OCR pixels back to PDF points
            # PDF Points = (OCR_Pixels / Scale_Factor)
            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height

            ocr_pages = FastPDFProcessor._perform_google_ocr(img_bytes)

            if ocr_pages:
                for ocr_page in ocr_pages:
                    for block in ocr_page.blocks:
                        # Extract text from paragraphs within the block
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = "".join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "

                        block_text = block_text.strip()
                        if not block_text: continue

                        # Calculate Bounding Box (un-scale it)
                        # Google returns vertices. We take min/max to form bbox.
                        xs = [v.x for v in block.bounding_box.vertices]
                        ys = [v.y for v in block.bounding_box.vertices]

                        if not xs or not ys: continue

                        x0, x1 = min(xs) / scale_x, max(xs) / scale_x
                        y0, y1 = min(ys) / scale_y, max(ys) / scale_y

                        chunks.append(TextBlock(
                            text=block_text,
                            bbox=(x0, y0, x1, y1),
                            page_num=page_num,
                            char_count=len(block_text),
                            source="ocr"
                        ))
        else:
            # Standard Processing logic (same as original)
            current_text = []
            current_bbox = list(raw_blocks[0][:4]) if raw_blocks else [0, 0, 0, 0]
            current_len = 0

            for block in raw_blocks:
                if block[6] != 0: continue  # Skip non-text
                text = block[4].strip()
                if not text: continue

                b_x0, b_y0, b_x1, b_y1 = block[:4]
                vertical_gap = b_y0 - current_bbox[3]

                is_new_section = vertical_gap > MERGE_TOLERANCE and len(current_text) > 0
                is_chunk_full = current_len > CHUNK_TARGET_SIZE

                if is_new_section or is_chunk_full:
                    full_text = " ".join(current_text)
                    if len(full_text) >= MIN_CHUNK_CHARS:
                        chunks.append(TextBlock(
                            text=full_text,
                            bbox=tuple(current_bbox),
                            page_num=page_num,
                            char_count=len(full_text),
                            source="native"
                        ))
                    current_text = [text]
                    current_bbox = [b_x0, b_y0, b_x1, b_y1]
                    current_len = len(text)
                else:
                    current_text.append(text)
                    current_len += len(text)
                    current_bbox[0] = min(current_bbox[0], b_x0)
                    current_bbox[1] = min(current_bbox[1], b_y0)
                    current_bbox[2] = max(current_bbox[2], b_x1)
                    current_bbox[3] = max(current_bbox[3], b_y1)

            if current_text:
                full_text = " ".join(current_text)
                if len(full_text) >= MIN_CHUNK_CHARS:
                    chunks.append(TextBlock(
                        text=full_text,
                        bbox=tuple(current_bbox),
                        page_num=page_num,
                        char_count=len(full_text),
                        source="native"
                    ))

        # --- Image Extraction (Standard) ---
        # Note: We skip this if OCR was triggered to save time,
        # or we can leave it enabled. Leaving enabled for completeness.
        image_list = page.get_images(full=True)
        img_counter = 0
        for img_info in image_list:
            try:
                xref = img_info[0]
                w, h = img_info[2], img_info[3]
                if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT: continue

                img_dict = doc.extract_image(xref)
                img_bytes = img_dict["image"]
                if len(img_bytes) < MIN_IMG_BYTES: continue

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
            ocr_performed=ocr_triggered
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
    # Ensure you have 'service-account-key.json' and 'docling.pdf'
    processor = FastPDFProcessor()
    result = processor.process_document("./ocr_pdf_test.pdf")
    processor.export_json(result)