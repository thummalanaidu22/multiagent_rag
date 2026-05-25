"""
DocumentIngestionAgent
======================
Objective : Process PDF documents, generate embeddings, and store them in
            ChromaDB for subsequent retrieval.
Input     : Path to a directory containing PDF files.
Output    : Populated ChromaDB collection + registered document records in
            long-term memory.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path

from utils.pdf_processor import PDFProcessor
from utils.embedder import Embedder
from memory.long_term import LongTermMemory
from config import VECTOR_STORE_PATH, COLLECTION_NAME, DOCUMENTS_DIR


class DocumentIngestionAgent:
    def __init__(self):
        self._processor = PDFProcessor()
        self._embedder = Embedder()
        self._ltm = LongTermMemory()
        self._client = chromadb.PersistentClient(
            path=VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

    def ingest(self, documents_dir: str = str(DOCUMENTS_DIR)) -> Dict[str, Any]:
        """
        Full ingestion pipeline:
        1. Parse PDFs → chunks
        2. Embed chunks
        3. Store in ChromaDB
        4. Register documents in long-term memory
        """
        pdf_files = list(Path(documents_dir).glob("*.pdf"))
        if not pdf_files:
            return {"status": "no_documents", "message": f"No PDFs found in {documents_dir}"}

        # Get or create collection (reset to re-ingest cleanly)
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        total_chunks = 0
        doc_stats = []

        for pdf_path in pdf_files:
            print(f"\n[IngestionAgent] Processing: {pdf_path.name}")
            chunks = self._processor.process_pdf(str(pdf_path))
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self._embedder.embed_batch(texts)

            ids = [f"{pdf_path.stem}_chunk_{c['chunk_id']}" for c in chunks]
            metadatas = [
                {
                    "source": c["source"],
                    "pages": str(c["pages"]),
                    "chunk_id": str(c["chunk_id"]),
                    "start_page": str(c["start_page"]),
                    "end_page": str(c["end_page"]),
                }
                for c in chunks
            ]

            # ChromaDB batch add (max 5000 per call)
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i: i + batch_size],
                    embeddings=embeddings[i: i + batch_size],
                    documents=texts[i: i + batch_size],
                    metadatas=metadatas[i: i + batch_size],
                )

            self._ltm.register_document(pdf_path.name, pdf_path.stem, len(chunks))
            total_chunks += len(chunks)
            doc_stats.append({"file": pdf_path.name, "chunks": len(chunks)})
            print(f"  → {len(chunks)} chunks indexed")

        result = {
            "status": "success",
            "documents_processed": len(doc_stats),
            "total_chunks": total_chunks,
            "details": doc_stats,
        }
        print(f"\n[IngestionAgent] Done. {len(doc_stats)} docs, {total_chunks} chunks indexed.")
        return result
