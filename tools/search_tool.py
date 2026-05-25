"""
SearchTool wraps ChromaDB vector search.
Used by the OrchestratorAgent and RetrieverAgent as a callable tool.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from utils.embedder import Embedder
from config import VECTOR_STORE_PATH, COLLECTION_NAME, TOP_K


class SearchTool:
    """
    Tool definition for Claude tool-use API:
        name: search_documents
        description: Search automotive standard documents for relevant content
    """

    TOOL_SCHEMA = {
        "name": "search_documents",
        "description": (
            "Search the automotive standards document corpus for chunks of text "
            "that are relevant to the given query. Returns the top-K most similar "
            "passages along with their source document and page numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to retrieve relevant document chunks for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5).",
                    "default": TOP_K,
                },
            },
            "required": ["query"],
        },
    }

    def __init__(self):
        self._embedder = Embedder()
        self._client = chromadb.PersistentClient(
            path=VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(COLLECTION_NAME)
            except Exception:
                self._collection = None
        return self._collection

    def execute(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Run vector similarity search and return top-k results."""
        collection = self._get_collection()
        if collection is None:
            return [{"error": "Vector store not initialised. Please run ingest.py first."}]

        query_embedding = self._embedder.embed(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "pages": meta.get("pages", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "score": round(1 - dist, 4),  # cosine similarity
            })
        return hits

    def collection_size(self) -> int:
        collection = self._get_collection()
        return collection.count() if collection else 0
