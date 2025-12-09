#!/usr/bin/env python3
"""
Chunk Visualizer - Draw bounding boxes on PDF files based on parsed document data.

Usage:
    python chunk_visualizer.py parsed_output.json document.pdf
    python chunk_visualizer.py parsed_output.json document.pdf --output annotated.pdf
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is required. Install with: pip install pymupdf")
    sys.exit(1)


@dataclass
class BoundingBox:
    left: float
    bottom: float
    right: float
    top: float
    page_num: int

    @classmethod
    def from_array(cls, bbox: List[float], page_num: int) -> "BoundingBox":
        """Create from [left, bottom, right, top] array format."""
        return cls(
            left=bbox[0],
            bottom=bbox[1],
            right=bbox[2],
            top=bbox[3],
            page_num=page_num
        )


@dataclass
class TextChunk:
    text: str
    bbox: BoundingBox
    source: str
    char_count: int


@dataclass
class ImageChunk:
    filename: str
    format: str
    bbox: BoundingBox
    width: int
    height: int


@dataclass
class ParsedPage:
    page_num: int
    texts: List[TextChunk]
    images: List[ImageChunk]


def load_parsed_document(json_path: str) -> Tuple[dict, List[ParsedPage]]:
    """Load parsed document from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("stats", {})
    pages = []

    for page_data in data.get("pages", []):
        page_num = page_data.get("page", 0)

        # Parse text chunks
        texts = []
        for text_item in page_data.get("text", []):
            bbox = BoundingBox.from_array(
                text_item["bbox"],
                text_item.get("page_num", page_num)
            )
            texts.append(TextChunk(
                text=text_item["text"],
                bbox=bbox,
                source=text_item.get("source", "unknown"),
                char_count=text_item.get("char_count", len(text_item["text"]))
            ))

        # Parse image chunks
        images = []
        for img_item in page_data.get("images", []):
            bbox = BoundingBox.from_array(
                img_item["bbox"],
                img_item.get("page_num", page_num)
            )
            images.append(ImageChunk(
                filename=img_item["filename"],
                format=img_item["format"],
                bbox=bbox,
                width=img_item["width"],
                height=img_item["height"]
            ))

        pages.append(ParsedPage(
            page_num=page_num,
            texts=texts,
            images=images
        ))

    return stats, pages


def bbox_to_rect(bbox: BoundingBox, page_rect: fitz.Rect) -> fitz.Rect:
    """
    Convert bounding box to PyMuPDF Rect.

    The bbox from pdf_processor is already in PyMuPDF's top-left origin format.
    BoundingBox field names are misleading: 'left' is x0, 'bottom' is y0,
    'right' is x1, 'top' is y1 (all in top-left origin).
    """
    # No conversion needed - bbox is already in PyMuPDF's coordinate system
    # Just map the fields directly: [x0, y0, x1, y1]
    return fitz.Rect(bbox.left, bbox.bottom, bbox.right, bbox.top)


# Color schemes
TEXT_COLORS = [
    (1.0, 0.9, 0.4),  # Yellow
    (0.6, 0.9, 0.6),  # Light green
    (0.7, 0.85, 1.0),  # Light blue
    (1.0, 0.8, 0.6),  # Peach
    (0.85, 0.75, 1.0),  # Lavender
    (0.6, 1.0, 0.9),  # Mint
    (1.0, 0.75, 0.8),  # Pink
    (0.9, 0.9, 0.7),  # Cream
]

IMAGE_COLORS = [
    (0.4, 0.7, 1.0),  # Blue
    (0.3, 0.8, 0.9),  # Cyan
    (0.5, 0.6, 1.0),  # Indigo
    (0.4, 0.9, 0.8),  # Teal
]


def get_text_color(index: int) -> Tuple[float, float, float]:
    """Get color for text chunk."""
    return TEXT_COLORS[index % len(TEXT_COLORS)]


def get_image_color(index: int) -> Tuple[float, float, float]:
    """Get color for image chunk."""
    return IMAGE_COLORS[index % len(IMAGE_COLORS)]


def make_text_label(text: str, max_word_len: int = 12) -> str:
    """
    Create a label from the first and last word of the text.

    Examples:
        "This is a text chunk" -> "This-chunk"
        "Hello" -> "Hello"
        "Hello World" -> "Hello-World"
    """
    # Clean and split into words
    words = text.split()

    if not words:
        return "T"

    # Get first word (truncate if too long)
    first = words[0].strip(".,;:!?\"'()[]{}").strip()
    if len(first) > max_word_len:
        first = first[:max_word_len]

    if len(words) == 1:
        return first

    # Get last word (truncate if too long)
    last = words[-1].strip(".,;:!?\"'()[]{}").strip()
    if len(last) > max_word_len:
        last = last[:max_word_len]

    # Avoid duplicates like "Hello-Hello"
    if first.lower() == last.lower():
        return first

    return f"{first}-{last}"


def draw_box_with_label(
        page: fitz.Page,
        rect: fitz.Rect,
        color: Tuple[float, float, float],
        label: str,
        opacity: float = 0.3,
        border_width: float = 1.5
) -> None:
    """Draw a highlighted box with a label."""
    # Draw filled rectangle
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(
        color=color,
        fill=color,
        fill_opacity=opacity,
        width=border_width
    )
    shape.commit()

    # Draw label background
    label_width = len(label) * 6 + 4
    label_rect = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x0 + label_width,
        rect.y0 + 12
    )

    # Ensure label stays within page bounds
    if label_rect.x1 > page.rect.width:
        label_rect.x0 = rect.x1 - label_width
        label_rect.x1 = rect.x1

    page.draw_rect(label_rect, color=(1, 1, 1), fill=(1, 1, 1))

    # Draw border around label for visibility
    darker_color = tuple(c * 0.7 for c in color)
    page.draw_rect(label_rect, color=darker_color, width=0.5)

    # Insert label text
    label_point = fitz.Point(label_rect.x0 + 2, label_rect.y0 + 9)
    page.insert_text(
        label_point,
        label,
        fontsize=8,
        color=(0, 0, 0),
        fontname="helv"
    )


def draw_chunks_on_pdf(
        pdf_path: str,
        pages: List[ParsedPage],
        output_path: str,
        show_labels: bool = True,
        opacity: float = 0.3,
        text_only: bool = False,
        images_only: bool = False
) -> dict:
    """Draw bounding boxes for all chunks on the PDF."""
    doc = fitz.open(pdf_path)
    stats = {"text_boxes": 0, "image_boxes": 0, "pages_annotated": 0}

    for parsed_page in pages:
        page_idx = parsed_page.page_num

        if page_idx >= len(doc):
            print(f"Warning: Page {page_idx} out of range (doc has {len(doc)} pages)")
            continue

        page = doc[page_idx]
        page_rect = page.rect
        page_has_annotations = False

        # Draw text chunks
        if not images_only:
            for idx, text_chunk in enumerate(parsed_page.texts):
                try:
                    rect = bbox_to_rect(text_chunk.bbox, page_rect)

                    # Skip invalid rectangles
                    if rect.is_empty or rect.is_infinite:
                        continue

                    color = get_text_color(idx)
                    label = make_text_label(text_chunk.text) if show_labels else ""

                    draw_box_with_label(page, rect, color, label, opacity)
                    stats["text_boxes"] += 1
                    page_has_annotations = True

                except Exception as e:
                    print(f"Warning: Could not draw text box {idx} on page {page_idx}: {e}")

        # Draw image chunks
        if not text_only:
            for idx, img_chunk in enumerate(parsed_page.images):
                try:
                    rect = bbox_to_rect(img_chunk.bbox, page_rect)

                    # Skip invalid rectangles
                    if rect.is_empty or rect.is_infinite:
                        continue

                    color = get_image_color(idx)
                    # Use filename without extension as label
                    img_label = Path(img_chunk.filename).stem
                    label = img_label if show_labels else ""

                    # Use slightly higher opacity and thicker border for images
                    draw_box_with_label(page, rect, color, label, opacity + 0.1, 2.0)
                    stats["image_boxes"] += 1
                    page_has_annotations = True

                except Exception as e:
                    print(f"Warning: Could not draw image box {idx} on page {page_idx}: {e}")

        if page_has_annotations:
            stats["pages_annotated"] += 1

    # Save the annotated PDF
    doc.save(output_path)
    doc.close()

    return stats


def print_summary(stats: dict, doc_stats: dict) -> None:
    """Print a summary of the visualization."""
    print("\n" + "=" * 50)
    print("VISUALIZATION SUMMARY")
    print("=" * 50)
    print(f"Document pages:     {doc_stats.get('total_pages', 'N/A')}")
    print(f"Pages annotated:    {stats['pages_annotated']}")
    print(f"Text boxes drawn:   {stats['text_boxes']}")
    print(f"Image boxes drawn:  {stats['image_boxes']}")
    print("=" * 50)
    print("\nLegend:")
    print("  Text chunks: 'FirstWord-LastWord' (yellow/green/blue tones)")
    print("  Image chunks: filename (blue/cyan tones)")
    print("=" * 50 + "\n")


def open_pdf(pdf_path: str) -> None:
    """Open the PDF with the system default viewer."""
    import subprocess
    import platform

    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", pdf_path], check=True)
        elif system == "Windows":
            subprocess.run(["start", "", pdf_path], shell=True, check=True)
        else:  # Linux
            subprocess.run(["xdg-open", pdf_path], check=True)
    except Exception as e:
        print(f"Could not open PDF automatically: {e}")
        print(f"Please open manually: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize parsed document chunks with bounding boxes on PDF"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for annotated PDF (default: <input>_annotated.pdf)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not show chunk labels"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.3,
        help="Opacity of highlight boxes (0.0-1.0, default: 0.3)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only visualize text chunks"
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only visualize image chunks"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the result"
    )

    args = parser.parse_args()

    # Validate inputs
    json_path = Path("./output.json")
    pdf_path = Path("./bafög_antwort.docx")

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Load parsed document
    print(f"Loading parsed data from: {json_path}")
    doc_stats, pages = load_parsed_document(str(json_path))
    print(f"Loaded {len(pages)} pages with {doc_stats.get('total_chunks', 'N/A')} chunks")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(pdf_path.with_stem(pdf_path.stem + "_annotated"))

    # Draw chunks
    print("Drawing bounding boxes...")
    viz_stats = draw_chunks_on_pdf(
        str(pdf_path),
        pages,
        output_path,
        show_labels=not args.no_labels,
        opacity=args.opacity,
        text_only=args.text_only,
        images_only=args.images_only
    )

    print(f"Saved annotated PDF to: {output_path}")
    print_summary(viz_stats, doc_stats)

    # Open result
    if not args.no_open:
        print("Opening annotated PDF...")
        open_pdf(output_path)


if __name__ == "__main__":
    main()