#!/usr/bin/env python3
"""
Step 2: Generate synthetic evaluation questions from ingested chunks.

Usage:
    python generate_questions.py
    python generate_questions.py --num-questions 50
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from chromadb.config import Settings
from evaluation.questionnaire import QuestionnaireGenerator
from config import VECTOR_STORE_PATH, COLLECTION_NAME, RESULTS_DIR


def load_chunks_from_vector_store() -> list:
    """Load all stored chunks from ChromaDB for question generation."""
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    total = collection.count()
    if total == 0:
        return []

    # Fetch all chunks (ChromaDB supports get() with limit)
    result = collection.get(
        limit=total,
        include=["documents", "metadatas"],
    )
    chunks = []
    for doc, meta, cid in zip(
        result["documents"], result["metadatas"], result["ids"]
    ):
        chunks.append({
            "chunk_id": meta.get("chunk_id", cid),
            "text": doc,
            "source": meta.get("source", ""),
            "pages": meta.get("pages", ""),
        })
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation questions.")
    parser.add_argument("--num-questions", type=int, default=50, help="Number of questions to generate")
    parser.add_argument("--output", default="questions.json", help="Output filename")
    args = parser.parse_args()

    print("Loading chunks from vector store...")
    chunks = load_chunks_from_vector_store()

    if not chunks:
        print("[ERROR] No chunks found. Please run ingest.py first.")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks. Generating {args.num_questions} questions...")
    gen = QuestionnaireGenerator()
    questions = gen.generate_from_chunks(chunks, total_questions=args.num_questions)
    path = gen.save(questions, filename=args.output)

    # Print a sample
    print("\n--- Sample Questions ---")
    for i, q in enumerate(questions[:5], 1):
        print(f"\n[{i}] [{q['difficulty'].upper()} / {q['type']}]")
        print(f"  Q: {q['question']}")
        print(f"  A: {q['answer'][:120]}...")
        print(f"  Source: {q.get('source', '')} | Pages: {q.get('pages', '')}")

    print(f"\nAll {len(questions)} questions saved to: {path}")


if __name__ == "__main__":
    main()
