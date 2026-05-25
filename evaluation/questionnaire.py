"""
QuestionnaireGenerator
======================
Uses Ollama LLM to generate diverse synthetic Q&A pairs from document chunks.
Produces questions across difficulty levels (easy/medium/hard) and
types (factual/reasoning/multi-hop).
"""
import json
import random
from openai import OpenAI
from typing import List, Dict, Any
from pathlib import Path

from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL, RESULTS_DIR

GENERATION_PROMPT = """You are creating evaluation questions for an automotive standards RAG system.

Given the excerpt below from an ARAI Automotive Industry Standard document, generate
{n} diverse questions with ground-truth answers.

Return ONLY a valid JSON array — no explanation, no markdown, no code fences.
Format:
[
  {{
    "question": "...",
    "answer": "...",
    "difficulty": "easy",
    "type": "factual",
    "source_hint": "brief phrase from context"
  }}
]

Difficulty: easy=single fact lookup, medium=rule/condition, hard=combine multiple facts
Type: factual=specific value/definition, reasoning=why/how, multi-hop=connect two facts

Document excerpt:
---
{context}
---

JSON array only:"""


class QuestionnaireGenerator:
    def __init__(self):
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    def generate_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        total_questions: int = 50,
        questions_per_chunk: int = 3,
    ) -> List[Dict[str, Any]]:
        questions: List[Dict[str, Any]] = []
        random.shuffle(chunks)

        for chunk in chunks:
            if len(questions) >= total_questions:
                break
            context = chunk.get("text", "")
            if len(context.split()) < 50:
                continue
            try:
                qa_list = self._generate_for_chunk(context, n=questions_per_chunk)
                for qa in qa_list:
                    qa["chunk_id"] = str(chunk.get("chunk_id", ""))
                    qa["source"]   = chunk.get("source", "")
                    qa["pages"]    = str(chunk.get("pages", ""))
                    questions.append(qa)
            except Exception as e:
                print(f"  [QuestionnaireGenerator] Skipped chunk: {e}")

        questions = self._balance(questions, total_questions)
        print(f"[QuestionnaireGenerator] Generated {len(questions)} questions.")
        return questions

    def _generate_for_chunk(self, context: str, n: int = 3) -> List[Dict[str, Any]]:
        prompt = GENERATION_PROMPT.format(context=context[:1800], n=n)
        response = self._client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model added them
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    raw = part
                    break
        # Find the JSON array bounds
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in model output")
        return json.loads(raw[start:end])

    @staticmethod
    def _balance(questions: List[Dict], n: int) -> List[Dict]:
        buckets: Dict[str, List] = {"easy": [], "medium": [], "hard": []}
        for q in questions:
            d = q.get("difficulty", "medium")
            buckets.setdefault(d, []).append(q)
        balanced = []
        per_bucket = n // 3
        for bucket in buckets.values():
            balanced.extend(bucket[:per_bucket])
        remaining = [q for q in questions if q not in balanced]
        balanced.extend(remaining[: n - len(balanced)])
        return balanced[:n]

    def save(self, questions: List[Dict[str, Any]], filename: str = "questions.json") -> str:
        path = RESULTS_DIR / filename
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"[QuestionnaireGenerator] Saved {len(questions)} questions → {path}")
        return str(path)

    @staticmethod
    def load(filename: str = "questions.json") -> List[Dict[str, Any]]:
        path = RESULTS_DIR / filename
        with open(path) as f:
            return json.load(f)
