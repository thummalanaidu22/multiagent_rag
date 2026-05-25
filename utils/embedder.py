"""
Embedding utility using sentence-transformers (local, no API key required).
Singleton pattern to avoid reloading the model on every call.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(f"Loading embedding model: {EMBEDDING_MODEL}")
            cls._instance._model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded.")
        return cls._instance

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings = self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )
        return embeddings.tolist()
