# Technical Design: Traceable Multi-Modal RAG System

## 1. Executive Summary
The objective is to build a high-performance Retrieval-Augmented Generation (RAG) system capable of ingesting multiple file types (PDF, DOCX, Images, etc.) with a specific focus on **tabular data** and **visual grounding**.

**Key Differentiators:**
1.  **Deep Traceability:** Users must be able to click a citation in the LLM response and be taken to the *exact* visual location (bounding box) in the source document.
2.  **Speed:** The ingestion pipeline targets a latency of ~5 seconds per document.
3.  **Layout Awareness:** High-fidelity parsing of complex structures like tables.

## 2. Why Docling?
[Docling](https://github.com/DS4SD/docling) is selected as the core ingestion engine because it solves the "PDF Black Box" problem. Unlike standard extractors (like PyPDF), Docling provides a hierarchical document representation including:
* **Provenance:** Precise bounding boxes (`bbox`) and page numbers for every paragraph, table cell, and image.
* **TableFormer:** State-of-the-art table structure recognition (preserving row/column headers).
* **Multi-format support:** Handles PDF, DOCX, PPTX, HTML, and Images seamlessly.

---

## 3. System Architecture

### A. Data Pipeline (Ingestion)
1.  **Upload:** User uploads file.
2.  **Docling Parsing:**
    * The file is processed using `DocumentConverter`.
    * *Optimization:* Attempt <5s target.
3.  **Chunking & Metadata Extraction:**
    * Instead of arbitrary character splitting, we chunk based on Docling's structural nodes (Paragraphs, List Items).
    * **Crucial:** We extract the `prov` (provenance/bbox) data for every chunk.
4.  **Embedding:** Chunks are embedded (e.g., via OpenAI, Cohere, or local BERT).
5.  **Storage:**
    * **Vector DB (e.g., Qdrant/Pinecone):** Stores Vectors + Payload (Text, FileID, PageNum, **JSON Bounding Box**).
    * **Object Store (S3/MinIO):** Stores the original PDF/Image file for frontend rendering.

### B. Retrieval & Generation
1.  **Semantic Search:** User query retrieves top $k$ chunks.
2.  **Context Construction:** The context fed to the LLM includes the text *and* a unique reference ID for each chunk.
3.  **Generation:** The LLM generates an answer, citing the specific Reference IDs.

### C. Frontend (The "Traceable" UI)
1.  **Rendering:** The frontend renders the markdown response.
2.  **Interaction:** When a user clicks a citation `[1]`:
    * Frontend fetches the `bbox` data associated with Chunk `[1]`.
    * A PDF Viewer (e.g., `react-pdf` or `pdf.js`) loads the source document.
    * A canvas overlay draws a highlight rectangle using the coordinates.

---


5. Risk Assessment & Optimization
The "5 Seconds" Constraint

Processing a multi-page PDF with OCR and Table Extraction in under 5 seconds is extremely aggressive. Deep-learning based layout analysis (which Docling uses) is compute-heavy.

Mitigation Strategies:

    Hybrid Parsing: * If the PDF is "digital-born" (not a scan), disable OCR in Docling options. This makes parsing 10x faster.

        Only enable OCR if no text is extracted in the first pass.

    Asynchronous UX:

        Even if backend takes 8 seconds, provide immediate feedback ("Analyzing layout...").

        Streaming: Embed and make chunks searchable as they are processed (page by page), rather than waiting for the whole doc.

The "Traceability" Complexity

    Issue: Text sometimes flows across pages.

    Solution: Docling handles this by providing a list of provenance items. If a paragraph splits across page 1 and 2, the prov array will contain two entries. Your frontend highlighter must handle drawing two separate boxes.

6. Conclusion

This architecture is feasible. Docling provides the exact data structure required for "Click-to-Source" functionality. The primary challenge will be tuning the infrastructure (GPU vs. CPU) to meet the latency requirements while maintaining the high quality of table extraction