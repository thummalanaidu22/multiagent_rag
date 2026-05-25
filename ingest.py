#!/usr/bin/env python3
"""
Step 1: Ingest PDF documents into the vector store.

Usage:
    python ingest.py
    python ingest.py --docs-dir /path/to/pdfs
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from agents.ingestion_agent import DocumentIngestionAgent
from config import DOCUMENTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into ChromaDB.")
    parser.add_argument(
        "--docs-dir",
        default=str(DOCUMENTS_DIR),
        help=f"Directory containing PDF files (default: {DOCUMENTS_DIR})",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"[ERROR] Directory not found: {docs_dir}")
        sys.exit(1)

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {docs_dir}")
        print("  → Please download 10-20 ARAI AIS documents and place them in:")
        print(f"     {docs_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to ingest:")
    for f in pdf_files:
        print(f"  • {f.name}")

    agent = DocumentIngestionAgent()
    result = agent.ingest(str(docs_dir))

    print("\n--- Ingestion Summary ---")
    print(f"  Documents processed : {result.get('documents_processed')}")
    print(f"  Total chunks indexed: {result.get('total_chunks')}")
    print("\nIngestion complete. You can now run main.py or generate_questions.py.")


if __name__ == "__main__":
    main()
