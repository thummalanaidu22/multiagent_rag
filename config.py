from pathlib import Path

BASE_DIR = Path(__file__).parent

# Ollama (local) — no API key needed
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"           # placeholder; Ollama ignores this
OLLAMA_MODEL    = "mistral:latest"   # used for orchestration + generation
OLLAMA_FAST_MODEL = "mistral:latest" # used for LLM-as-judge (lightweight calls)

# Paths
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_STORE_PATH = str(BASE_DIR / "vector_store")
LONG_TERM_DB_PATH = str(BASE_DIR / "memory" / "long_term.db")
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# ChromaDB
COLLECTION_NAME = "automotive_standards"

# Embedding model (local, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Path(LONG_TERM_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
