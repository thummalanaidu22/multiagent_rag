#!/usr/bin/env python3
"""
Step 3: Run full evaluation over the synthetic question set.

For each question:
  - Retrieve top-K chunks
  - Generate an answer
  - Compute Precision@K, Recall@K, MRR (using source chunk as ground truth)
  - Compute Faithfulness and Answer Relevance via LLM-as-judge
  - Save all results

Usage:
    python evaluate.py
    python evaluate.py --questions evaluation/results/questions.json --k 5
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from evaluation.metrics import compute_metrics_report
from evaluation.questionnaire import QuestionnaireGenerator
from config import TOP_K, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument(
        "--questions",
        default=str(RESULTS_DIR / "questions.json"),
        help="Path to questions JSON file",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help=f"Top-K for retrieval (default {TOP_K})")
    parser.add_argument("--output-prefix", default="evaluation", help="Prefix for output files")
    args = parser.parse_args()

    # Load questions
    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"[ERROR] Questions file not found: {qpath}")
        print("  → Run generate_questions.py first.")
        sys.exit(1)

    with open(qpath) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions.")

    retriever = RetrieverAgent()
    generator = GeneratorAgent()
    evaluator = EvaluatorAgent(k=args.k)

    records = []
    print(f"\nRunning evaluation (K={args.k})...")

    for q in tqdm(questions, desc="Evaluating"):
        query = q["question"]
        expected_answer = q.get("answer", "")
        ground_truth_chunk_id = str(q.get("chunk_id", ""))

        # Retrieve
        chunks = retriever.retrieve(query, k=args.k)
        retrieved_ids = [str(c.get("chunk_id", "")) for c in chunks]
        context_texts = [c["text"] for c in chunks]

        # Relevant IDs = the source chunk used to generate this question
        relevant_ids = [ground_truth_chunk_id] if ground_truth_chunk_id else []

        # Generate answer
        answer = generator.generate(query, chunks)

        records.append({
            "query": query,
            "expected_answer": expected_answer,
            "answer": answer,
            "context": context_texts,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "sources": [c["source"] for c in chunks],
            "difficulty": q.get("difficulty", ""),
            "type": q.get("type", ""),
        })

    print("\nComputing metrics...")
    metrics = evaluator.full_evaluation(records)

    # Also break down by difficulty
    print("\n--- Metrics by Difficulty ---")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in records if r.get("difficulty") == diff]
        if subset:
            sub_metrics = evaluator.full_evaluation(subset)
            faith = sub_metrics.get("faithfulness", 0)
            rel = sub_metrics.get("answer_relevance", 0)
            p = sub_metrics.get(f"precision@{args.k}", 0)
            print(f"  {diff:8s}: Precision@{args.k}={p:.3f}  Faith={faith:.3f}  Relevance={rel:.3f}  (n={len(subset)})")

    compute_metrics_report(records, metrics, k=args.k, output_prefix=args.output_prefix)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
