"""
EvaluatorAgent
==============
Objective : Evaluate retrieval and generation quality using standard metrics.
Input     : List of (query, retrieved_chunk_ids, relevant_chunk_ids, answer, context) tuples.
Output    : Precision@K, Recall@K, MRR, Faithfulness, Answer Relevance scores.
"""
import json
from typing import List, Dict, Any
from tools.eval_tool import EvalTool
from agents.generator_agent import GeneratorAgent
from config import TOP_K


class EvaluatorAgent:
    def __init__(self, k: int = TOP_K):
        self.k = k
        self._eval_tool = EvalTool()
        self._generator = GeneratorAgent()

    def evaluate_retrieval(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute Precision@K, Recall@K, MRR over a list of evaluation records.
        Each record must have: retrieved_ids (list), relevant_ids (list).
        """
        p_scores, r_scores, mrr_scores = [], [], []

        for rec in records:
            retrieved = rec.get("retrieved_ids", [])
            relevant = rec.get("relevant_ids", [])
            if not relevant:
                continue
            p_scores.append(EvalTool.precision_at_k(retrieved, relevant, self.k))
            r_scores.append(EvalTool.recall_at_k(retrieved, relevant, self.k))
            mrr_scores.append(EvalTool.mrr(retrieved, relevant))

        n = len(p_scores) or 1
        return {
            f"precision@{self.k}": round(sum(p_scores) / n, 4),
            f"recall@{self.k}": round(sum(r_scores) / n, 4),
            "mrr": round(sum(mrr_scores) / n, 4),
            "num_queries": n,
        }

    def evaluate_generation(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute average Faithfulness and Answer Relevance using LLM-as-judge.
        Each record must have: query, answer, context (list of str).
        """
        faith_scores, rel_scores = [], []

        for rec in records:
            query = rec.get("query", "")
            answer = rec.get("answer", "")
            context = "\n\n".join(rec.get("context", []))

            faith = self._generator.judge_faithfulness(query, answer, context)
            rel = self._generator.judge_relevance(query, answer)

            faith_scores.append(faith)
            rel_scores.append(rel)

            rec["faithfulness"] = faith
            rec["answer_relevance"] = rel

        n = len(faith_scores) or 1
        return {
            "faithfulness": round(sum(faith_scores) / n, 4),
            "answer_relevance": round(sum(rel_scores) / n, 4),
            "num_evaluated": n,
        }

    def full_evaluation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run both retrieval and generation evaluation and merge results."""
        retrieval_metrics = self.evaluate_retrieval(records)
        generation_metrics = self.evaluate_generation(records)
        return {**retrieval_metrics, **generation_metrics}
