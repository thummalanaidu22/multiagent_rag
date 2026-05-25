"""
EvalTool computes retrieval and generation quality metrics.
Exposed as a tool schema for use by the EvaluatorAgent.
"""
from typing import List, Dict, Any


class EvalTool:
    SCHEMA = {
        "name": "evaluate_response",
        "description": (
            "Evaluate the quality of a generated answer against retrieved context. "
            "Returns faithfulness and answer_relevance scores (0-1)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The original user question."},
                "answer": {"type": "string", "description": "The generated answer to evaluate."},
                "context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of retrieved context chunks used to generate the answer.",
                },
            },
            "required": ["query", "answer", "context"],
        },
    }

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Fraction of top-K retrieved chunks that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_set = set(relevant_ids)
        hits = sum(1 for cid in top_k if cid in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Fraction of all relevant chunks that appear in top-K."""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        hits = sum(1 for cid in relevant_ids if cid in top_k)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        relevant_set = set(relevant_ids)
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_all(
        retrieved_ids: List[str], relevant_ids: List[str], k: int
    ) -> Dict[str, float]:
        return {
            f"precision@{k}": EvalTool.precision_at_k(retrieved_ids, relevant_ids, k),
            f"recall@{k}": EvalTool.recall_at_k(retrieved_ids, relevant_ids, k),
            "mrr": EvalTool.mrr(retrieved_ids, relevant_ids),
        }
