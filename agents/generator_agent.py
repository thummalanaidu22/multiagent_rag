"""
GeneratorAgent
==============
Objective : Generate a grounded, accurate answer using retrieved context + Ollama LLM.
Input     : query (str), context_chunks (List[Dict]), conversation history.
Output    : Generated answer string.
"""
from openai import OpenAI
from typing import List, Dict, Any
from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_FAST_MODEL

SYSTEM_PROMPT = """You are an expert assistant specialising in Automotive Industry Standards (AIS)
published by ARAI (Automotive Research Association of India).

Rules:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say so clearly.
3. Cite the source document and page number for every factual claim.
4. Be precise and technical. Use exact values, units, and thresholds from the documents.
5. For multi-part questions, structure your answer with numbered points.
"""


class GeneratorAgent:
    def __init__(self):
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        context_text = self._format_context(context_chunks)
        user_message = (
            f"Context from automotive standards documents:\n\n"
            f"{context_text}\n\n---\n\n"
            f"Question: {query}\n\n"
            f"Answer based strictly on the context above, citing sources."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history or [])
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No context available."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i}] Source: {c.get('source', 'unknown')} "
                f"| Pages: {c.get('pages', 'N/A')}\n"
                f"{c.get('text', '')}"
            )
        return "\n\n".join(parts)

    def judge_faithfulness(self, query: str, answer: str, context: str) -> float:
        prompt = (
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {query}\nAnswer: {answer}\n\n"
            "Rate how faithfully the answer is supported by the context alone. "
            "Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
        )
        resp = self._client.chat.completions.create(
            model=OLLAMA_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def judge_relevance(self, query: str, answer: str) -> float:
        prompt = (
            f"Question: {query}\nAnswer: {answer}\n\n"
            "Rate how directly and completely the answer addresses the question. "
            "Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
        )
        resp = self._client.chat.completions.create(
            model=OLLAMA_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except ValueError:
            return 0.5
