"""
RetrieverAgent
==============
Objective : Given a natural-language query, retrieve the most relevant document
            chunks from the vector store.
Input     : query (str), optional k (int)
Output    : List of ranked chunks with text, source, page numbers, and score.
"""
from typing import List, Dict, Any
from tools.search_tool import SearchTool
from config import TOP_K


class RetrieverAgent:
    def __init__(self):
        self._search = SearchTool()

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        Applies a minimum similarity threshold to filter noise.
        """
        results = self._search.execute(query, k=k)

        # Filter results with error keys
        if results and "error" in results[0]:
            print(f"[RetrieverAgent] {results[0]['error']}")
            return []

        # Soft filter: drop chunks below 0.2 similarity (very low relevance)
        filtered = [r for r in results if r.get("score", 0) >= 0.2]
        if not filtered:
            filtered = results  # fall back to all results if everything is below threshold

        return filtered

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Render retrieved chunks as a numbered context block for the LLM."""
        if not chunks:
            return "No relevant context found."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {c['source']} | Pages: {c['pages']}\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def collection_info(self) -> Dict[str, Any]:
        size = self._search.collection_size()
        return {"collection": "automotive_standards", "total_chunks": size}
