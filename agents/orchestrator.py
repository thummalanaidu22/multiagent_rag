"""
OrchestratorAgent
=================
Objective : Coordinate the full RAG pipeline using Ollama tool-use.
            Drives a tool-calling loop: search → (memory check) → generate.
Input     : User query (str).
Output    : Answer (str) + retrieved chunks.

Tool-calling strategy:
  Small local models can be inconsistent with native tool calling, so we use a
  deterministic two-phase approach:
    Phase 1 — always call search_documents (retrieval is mandatory for RAG).
    Phase 2 — optionally call retrieve_from_memory for follow-ups.
    Phase 3 — send retrieved context to Ollama and generate the final answer.
  The tools are defined in OpenAI tool-use format and logged for transparency.
"""
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from tools.search_tool import SearchTool
from tools.memory_tool import MemoryTool
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL, TOP_K

SYSTEM_PROMPT = """You are an OrchestratorAgent for an Automotive Standards RAG chatbot.
You answer questions about ARAI Automotive Industry Standards (AIS) documents.

You have two tools:
- search_documents: search the document corpus for relevant passages
- retrieve_from_memory: look up previously answered questions

For EVERY question, you MUST call search_documents first.
After getting results, synthesise a precise, cited answer.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search automotive standard documents for relevant content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k":     {"type": "integer", "description": "Number of results", "default": TOP_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_memory",
            "description": "Look up past answered questions matching a keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search"},
                    "limit":   {"type": "integer", "default": 3},
                },
                "required": ["keyword"],
            },
        },
    },
]


class OrchestratorAgent:
    def __init__(self):
        self._client    = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        self._retriever = RetrieverAgent()
        self._generator = GeneratorAgent()
        self._search    = SearchTool()
        self._memory    = MemoryTool()
        self._stm       = ShortTermMemory()
        self._ltm       = LongTermMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_query: str) -> Tuple[str, List[Dict]]:
        """Process a query through the tool-use loop. Returns (answer, chunks)."""
        self._stm.add_turn("user", user_query)
        retrieved_chunks: List[Dict] = []

        # Build message history for the model
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._stm.get_messages()[:-1])  # history except current
        messages.append({"role": "user", "content": user_query})

        # Agentic loop (max 4 iterations)
        for _ in range(4):
            response = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message

            # No tool calls → final answer
            if not msg.tool_calls:
                answer = msg.content or "No answer generated."
                break

            # Execute tool calls
            messages.append(msg)
            for tc in msg.tool_calls:
                result, chunks = self._dispatch(tc.function.name, tc.function.arguments)
                retrieved_chunks.extend(chunks)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        else:
            # Fallback: if loop exhausted, force-generate with whatever we have
            answer = self._fallback_generate(user_query, retrieved_chunks)

        # If model never returned text (only tool calls), generate explicitly
        if not msg.tool_calls and not (msg.content or "").strip():
            answer = self._fallback_generate(user_query, retrieved_chunks)

        self._stm.add_turn("assistant", answer)
        sources = list({c["source"] for c in retrieved_chunks})
        self._ltm.save_interaction(
            session_id=self._stm.session_id,
            query=user_query,
            answer=answer,
            context=[c["text"] for c in retrieved_chunks],
            sources=sources,
        )
        return answer, retrieved_chunks

    def reset_conversation(self):
        self._stm.clear()

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, args_json: str) -> Tuple[Any, List[Dict]]:
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {}

        if name == "search_documents":
            query = args.get("query", "")
            k     = args.get("k", TOP_K)
            chunks = self._retriever.retrieve(query, k=k)
            result = [{"text": c["text"], "source": c["source"],
                       "pages": c["pages"], "score": c["score"]} for c in chunks]
            return result, chunks

        if name == "retrieve_from_memory":
            keyword = args.get("keyword", "")
            limit   = args.get("limit", 3)
            return self._memory.retrieve(keyword, limit=limit), []

        return {"error": f"Unknown tool: {name}"}, []

    def _fallback_generate(self, query: str, chunks: List[Dict]) -> str:
        """Called when the tool loop ends without a text answer."""
        return self._generator.generate(query, chunks, conversation_history=[])
