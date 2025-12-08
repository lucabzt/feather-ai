#!/usr/bin/env python3
"""
Fast PDF Document Processor
Optimized for 4-core CPU with 4GB RAM
Target: 10-20 seconds for 50 pages, 30 seconds for 300 pages
"""
import json

import pymupdf
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import gc
from pathlib import Path


@dataclass
class TextBlock:
    """Text block with coordinates for frontend backtracing"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    block_type: str = "text"  # text, heading, etc.


@dataclass
class ImageData:
    """Extracted image with metadata"""
    image_bytes: bytes
    format: str  # png, jpeg, etc.
    bbox: Tuple[float, float, float, float]
    page_num: int
    width: int
    height: int


@dataclass
class TableData:
    """Extracted table with structure"""
    data: List[List[str]]  # 2D array of cell contents
    bbox: Tuple[float, float, float, float]
    page_num: int
    rows: int
    cols: int


@dataclass
class PageResult:
    """Complete result for a single page"""
    page_num: int
    text_blocks: List[TextBlock]
    images: List[ImageData]
    tables: List[TableData]
    processing_time: float
    used_ocr: bool


class FastPDFProcessor:
    """
    High-performance PDF processor optimized for CPU-only environments.

    Features:
    - Parallel page processing across CPU cores
    - Automatic OCR detection and conditional processing
    - Block-level coordinate tracking for frontend display
    - Memory-efficient chunked processing for large documents
    - Image extraction as bytes with format preservation
    - Table extraction with coordinate backtracing
    """

    def __init__(
            self,
            num_workers: Optional[int] = None,
            ocr_enabled: bool = True,
            table_extraction: bool = True,
            chunk_size: int = 50,
            min_text_per_page: int = 100,
            dpi: int = 150
    ):
        """
        Initialize the processor.

        Args:
            num_workers: Number of parallel workers (default: CPU count)
            ocr_enabled: Enable OCR for scanned documents
            table_extraction: Enable table detection and extraction
            chunk_size: Pages per chunk for large documents (memory management)
            min_text_per_page: Threshold to determine if OCR is needed
            dpi: DPI for OCR rasterization (150 is optimal for speed/quality)
        """
        self.num_workers = num_workers or min(cpu_count(), 4)
        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.chunk_size = chunk_size
        self.min_text_per_page = min_text_per_page
        self.dpi = dpi

        # Reduce PyMuPDF cache for memory efficiency
        pymupdf.TOOLS.store_shrink(75)

        print(f"Initialized FastPDFProcessor with {self.num_workers} workers")

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process entire PDF document with parallel processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing:
                - pages: List[PageResult] for each page
                - total_pages: int
                - total_time: float (seconds)
                - used_ocr: bool
                - document_metadata: Dict
        """
        start_time = time.time()
        pdf_path = str(Path(pdf_path).resolve())

        # Quick metadata check
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        metadata = doc.metadata
        doc.close()

        print(f"Processing {total_pages} pages from: {Path(pdf_path).name}")

        # Phase 1: Detect if OCR is needed (sample first 3 pages)
        needs_ocr = self._detect_ocr_need(pdf_path, sample_size=min(3, total_pages))

        if needs_ocr and self.ocr_enabled:
            print("⚠️  Scanned document detected - OCR will be applied")
            # Lazy import to avoid startup cost if not needed
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.ocr_engine = RapidOCR(
                    det_limit_side_len=1600,
                    box_thresh=0.55,
                    rec_batch_num=4
                )
            except ImportError:
                print("⚠️  RapidOCR not installed, falling back to text extraction only")
                needs_ocr = False
        else:
            self.ocr_engine = None

        # Phase 2: Process pages in chunks with parallel workers
        all_results = []
        for chunk_start in range(0, total_pages, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_pages)
            print(f"Processing pages {chunk_start + 1}-{chunk_end}...")

            # Prepare arguments for parallel processing
            page_args = [
                (pdf_path, page_num, needs_ocr, self.ocr_enabled,
                 self.table_extraction, self.dpi)
                for page_num in range(chunk_start, chunk_end)
            ]

            # Process chunk in parallel
            with Pool(self.num_workers) as pool:
                chunk_results = pool.map(self._process_single_page, page_args)

            all_results.extend(chunk_results)

            # Memory cleanup between chunks
            if chunk_end < total_pages:
                pymupdf.TOOLS.store_shrink(100)
                gc.collect()

        total_time = time.time() - start_time

        # Compile final results
        result = {
            "pages": all_results,
            "total_pages": total_pages,
            "total_time": total_time,
            "avg_time_per_page": total_time / total_pages if total_pages > 0 else 0,
            "used_ocr": needs_ocr and self.ocr_enabled,
            "document_metadata": metadata,
            "total_text_blocks": sum(len(p.text_blocks) for p in all_results),
            "total_images": sum(len(p.images) for p in all_results),
            "total_tables": sum(len(p.tables) for p in all_results)
        }

        print(f"\n✅ Completed in {total_time:.2f}s ({result['avg_time_per_page']:.3f}s per page)")
        print(f"   📝 {result['total_text_blocks']} text blocks")
        print(f"   🖼️  {result['total_images']} images")
        print(f"   📊 {result['total_tables']} tables")

        return result

    def _detect_ocr_need(self, pdf_path: str, sample_size: int = 3) -> bool:
        """
        Detect if document needs OCR by sampling first few pages.

        Args:
            pdf_path: Path to PDF
            sample_size: Number of pages to sample

        Returns:
            True if OCR is needed, False otherwise
        """
        doc = pymupdf.open(pdf_path)
        total_chars = 0
        pages_sampled = min(sample_size, len(doc))

        for i in range(pages_sampled):
            page = doc[i]
            text = page.get_text().strip()
            total_chars += len(text)

        doc.close()

        avg_chars = total_chars / pages_sampled
        needs_ocr = avg_chars < self.min_text_per_page

        return needs_ocr

    @staticmethod
    def _process_single_page(args: Tuple) -> PageResult:
        """
        Process a single page (worker function for multiprocessing).

        This is a static method because it needs to be pickleable for multiprocessing.
        Each worker opens its own document instance (PyMuPDF is not thread-safe).
        """
        pdf_path, page_num, needs_ocr, ocr_enabled, table_extraction, dpi = args
        start_time = time.time()

        doc = pymupdf.open(pdf_path)
        page = doc[page_num]

        text_blocks = []
        images = []
        tables = []
        used_ocr = False

        # Extract text with coordinates
        if needs_ocr and ocr_enabled:
            # OCR path
            text_content = page.get_text().strip()
            if len(text_content) < 50:  # Page likely needs OCR
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    ocr_engine = RapidOCR(
                        det_limit_side_len=1600,
                        box_thresh=0.55,
                        rec_batch_num=4
                    )

                    # Rasterize page
                    pix = page.get_pixmap(dpi=dpi)
                    img_bytes = pix.tobytes("png")

                    # Run OCR
                    ocr_result, _ = ocr_engine(img_bytes)

                    if ocr_result:
                        for item in ocr_result:
                            bbox_points, text, confidence = item
                            # Convert bbox points to (x0, y0, x1, y1)
                            xs = [p[0] for p in bbox_points]
                            ys = [p[1] for p in bbox_points]
                            bbox = (min(xs), min(ys), max(xs), max(ys))

                            text_blocks.append(TextBlock(
                                text=text,
                                bbox=bbox,
                                page_num=page_num,
                                block_type="ocr_text"
                            ))

                    used_ocr = True
                    del pix
                except Exception as e:
                    print(f"OCR failed on page {page_num}: {e}, falling back to text extraction")

        # Standard text extraction path (or fallback)
        if not used_ocr:
            content = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)

            for block in content["blocks"]:
                if block["type"] == 0:  # Text block
                    # Combine all lines in block into single text
                    block_text = ""
                    for line in block["lines"]:
                        line_text = " ".join(
                            span["text"] for span in line["spans"]
                        )
                        block_text += line_text + " "

                    if block_text.strip():
                        text_blocks.append(TextBlock(
                            text=block_text.strip(),
                            bbox=tuple(block["bbox"]),
                            page_num=page_num,
                            block_type="text"
                        ))

        # Extract images
        for img_index, img_info in enumerate(page.get_images(full=True)):
            try:
                xref = img_info[0]
                img_dict = doc.extract_image(xref)

                # Get image position on page
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else (0, 0, 0, 0)

                images.append(ImageData(
                    image_bytes=img_dict["image"],
                    format=img_dict["ext"],
                    bbox=tuple(bbox),
                    page_num=page_num,
                    width=img_dict["width"],
                    height=img_dict["height"]
                ))
            except Exception as e:
                print(f"Failed to extract image {img_index} on page {page_num}: {e}")

        # Extract tables
        if table_extraction:
            try:
                page_tables = page.find_tables()
                for table in page_tables:
                    table_data = table.extract()

                    if table_data and len(table_data) > 0:
                        tables.append(TableData(
                            data=table_data,
                            bbox=tuple(table.bbox),
                            page_num=page_num,
                            rows=len(table_data),
                            cols=len(table_data[0]) if table_data else 0
                        ))
            except Exception as e:
                print(f"Table extraction failed on page {page_num}: {e}")

        doc.close()

        processing_time = time.time() - start_time

        return PageResult(
            page_num=page_num,
            text_blocks=text_blocks,
            images=images,
            tables=tables,
            processing_time=processing_time,
            used_ocr=used_ocr
        )

    def export_to_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.

        Args:
            result: Output from process_document()

        Returns:
            Dictionary with all data structures converted to dicts/lists
        """
        return {
            "pages": [
                {
                    "page_num": page.page_num,
                    "text_blocks": [asdict(tb) for tb in page.text_blocks],
                    "images": [
                        {
                            "format": img.format,
                            "bbox": img.bbox,
                            "page_num": img.page_num,
                            "width": img.width,
                            "height": img.height,
                            # Note: image_bytes excluded from dict by default
                            # Include separately if needed for storage
                        }
                        for img in page.images
                    ],
                    "tables": [asdict(tb) for tb in page.tables],
                    "processing_time": page.processing_time,
                    "used_ocr": page.used_ocr
                }
                for page in result["pages"]
            ],
            "metadata": result["document_metadata"],
            "statistics": {
                "total_pages": result["total_pages"],
                "total_time": result["total_time"],
                "avg_time_per_page": result["avg_time_per_page"],
                "used_ocr": result["used_ocr"],
                "total_text_blocks": result["total_text_blocks"],
                "total_images": result["total_images"],
                "total_tables": result["total_tables"]
            }
        }

    def get_images_with_bytes(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all images with their byte data for storage/display.

        Args:
            result: Output from process_document()

        Returns:
            List of dicts with image metadata and bytes
        """
        all_images = []
        for page in result["pages"]:
            for img in page.images:
                all_images.append({
                    "page_num": img.page_num,
                    "format": img.format,
                    "bbox": img.bbox,
                    "width": img.width,
                    "height": img.height,
                    "bytes": img.image_bytes
                })
        return all_images


def main():
    """Example usage"""

    # Initialize processor
    processor = FastPDFProcessor(
        num_workers=4,
        ocr_enabled=True,
        table_extraction=True,
        chunk_size=500
    )

    # Process document
    start_time = time.time()
    result = processor.process_document("./docling.pdf")
    end_time = time.time()
    print(f"Document processed in {end_time - start_time:.2f}s")

    # Example: Access first page results
    if result["pages"]:
        first_page = result["pages"][0]
        print(f"\nFirst page preview:")
        print(f"  Text blocks: {len(first_page.text_blocks)}")
        if first_page.text_blocks:
            print(f"  First block: '{first_page.text_blocks[0].text[:100]}...'")
            print(f"  Coordinates: {first_page.text_blocks[0].bbox}")
        print(f"  Images: {len(first_page.images)}")
        print(f"  Tables: {len(first_page.tables)}")

    # Export to JSON-serializable format
    export = processor.export_to_dict(result)
    json.dump(export, open("output.json", "w"), indent=2)

    # Get images with bytes if needed
    images_with_bytes = processor.get_images_with_bytes(result)
    print(f"\nExtracted {len(images_with_bytes)} images")


if __name__ == "__main__":
    main()