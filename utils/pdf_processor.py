"""
PDF parsing and chunking utilities.
Uses PyMuPDF for robust extraction of technical/unstructured PDFs.
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text page-by-page from a PDF."""
        pages = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                pages.append({
                    "page": page_num,
                    "text": text,
                    "source": Path(pdf_path).name,
                })
        doc.close()
        return pages

    def _clean_text(self, text: str) -> str:
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page header/footer artefacts (short lines with only digits/dashes)
        lines = [l for l in text.split("\n") if not re.match(r"^\s*[\d\-–—]+\s*$", l)]
        return "\n".join(lines).strip()

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge all page text then split into overlapping chunks by word count.
        Each chunk records its source file and approximate page range.
        """
        chunks = []
        # Build a flat list of (word, page_num, source)
        word_meta: List[tuple] = []
        for page in pages:
            words = page["text"].split()
            for w in words:
                word_meta.append((w, page["page"], page["source"]))

        start = 0
        chunk_id = 0
        while start < len(word_meta):
            end = min(start + self.chunk_size, len(word_meta))
            segment = word_meta[start:end]
            text = " ".join(w for w, _, _ in segment)
            pages_covered = sorted(set(p for _, p, _ in segment))
            source = segment[0][2]
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "pages": pages_covered,
                "start_page": pages_covered[0],
                "end_page": pages_covered[-1],
            })
            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Full pipeline: extract → clean → chunk."""
        pages = self.extract_text(pdf_path)
        return self.chunk_pages(pages)

    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory."""
        all_chunks = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF(s) in {directory}")
        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")
            chunks = self.process_pdf(str(pdf_path))
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        return all_chunks
