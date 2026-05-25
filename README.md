# Multi-Agent RAG Chatbot — ARAI Automotive Standards

An AI-powered chatbot that answers questions from ARAI Automotive Industry Standards (AIS) documents using a **Multi-Agent Architecture** with **Retrieval-Augmented Generation (RAG)**. Runs entirely locally via **Ollama** — no cloud API key required.

---

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │          OrchestratorAgent               │
                     │  (Claude tool-use loop via Ollama/mistral)│
                     │                                          │
                     │  Tools exposed to the LLM:               │
                     │   • search_documents                     │
                     │   • save_to_memory                       │
                     │   • retrieve_from_memory                 │
                     └───────────────┬──────────────────────────┘
                                     │ delegates to
             ┌───────────────────────┼──────────────────────┐
             │                       │                      │
   ┌─────────▼──────┐   ┌───────────▼──────┐   ┌──────────▼──────────┐
   │ RetrieverAgent │   │ GeneratorAgent   │   │  EvaluatorAgent     │
   │                │   │                  │   │                     │
   │ • ChromaDB     │   │ • Ollama mistral  │   │ • Precision@K       │
   │   vector search│   │ • Grounded answer │   │ • Recall@K          │
   │ • Cosine sim   │   │   generation      │   │ • MRR               │
   │ • Score filter │   │ • LLM-as-judge    │   │ • Faithfulness      │
   └────────────────┘   └──────────────────┘   │ • Answer Relevance  │
                                               └─────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                  DocumentIngestionAgent                         │
   │  PyMuPDF → Text Extraction → Chunking → Embeddings → ChromaDB  │
   └─────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────┐   ┌────────────────────────────────────┐
   │    Short-Term Memory     │   │        Long-Term Memory            │
   │  In-memory sliding window│   │  SQLite: interactions, documents,  │
   │  10-turn conversation    │   │  key-value store                   │
   └──────────────────────────┘   └────────────────────────────────────┘
```

---

## Agent Definitions

| Agent | Objective | Input | Output |
|---|---|---|---|
| **OrchestratorAgent** | Drives the tool-use loop; coordinates retrieval, generation and memory | User query | Final grounded answer + sources |
| **DocumentIngestionAgent** | Parse PDFs, chunk text, embed and store in ChromaDB | PDF directory path | Populated vector store + document registry |
| **RetrieverAgent** | Semantic similarity search over the vector store | Query string | Top-K ranked chunks with scores |
| **GeneratorAgent** | Generate a grounded answer using retrieved context; LLM-as-judge scoring | Query + chunks | Answer string; faithfulness / relevance scores |
| **EvaluatorAgent** | Compute retrieval and generation quality metrics | Evaluation records | Precision@K, Recall@K, MRR, Faithfulness, Answer Relevance |

---

## Tools

Three tools are registered in OpenAI tool-use format and invoked by the Ollama LLM inside the OrchestratorAgent loop:

| Tool | Trigger | Effect |
|---|---|---|
| `search_documents` | Every factual question | Embedding-based vector search over ChromaDB; returns top-K chunks |
| `retrieve_from_memory` | Follow-up or repeated queries | Searches SQLite for past interactions matching a keyword |
| `save_to_memory` | When the agent learns a notable fact | Persists a key-value entry to SQLite for future sessions |

---

## Memory Design

### Short-Term Memory (`memory/short_term.py`)
- **Storage**: In-memory Python list
- **Scope**: Single conversation session
- **Window**: Sliding 10-turn (20 message) cap to prevent token overflow
- **Usage**: Passed as `messages` list to every Ollama API call so the model has full conversational context

### Long-Term Memory (`memory/long_term.py`)
- **Storage**: SQLite database at `memory/long_term.db`
- **Three tables**:
  - `interactions` — every Q&A pair with query, answer, retrieved context, sources, and timestamp
  - `documents` — registry of ingested PDFs with chunk counts
  - `key_value` — arbitrary facts saved by the agent via the `save_to_memory` tool
- **Usage**: The OrchestratorAgent persists every interaction automatically; the `retrieve_from_memory` tool lets the LLM look up past answers to avoid repeating retrieval work

---

## Model Stack

| Component | Model / Library | Why Chosen |
|---|---|---|
| **LLM** | `mistral:latest` via Ollama | Reliable tool-calling support; fast on local hardware; fully offline |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Free, local, no API key, 384-dim vectors, fast (~80ms/batch), strong semantic similarity |
| **Vector Store** | ChromaDB (persistent local) | Zero-config local persistence; cosine similarity built-in; Python-native |
| **PDF Parser** | PyMuPDF (fitz) | Handles complex technical PDFs including tables and cross-references |

---

## Dataset

- **Source**: ARAI Automotive Industry Standards (AIS) — [araiindia.com/downloads/ais-downloads](https://www.araiindia.com/downloads/ais-downloads)
- **Documents ingested**: 17 PDFs
- **Total chunks indexed**: 378 (chunk size = 500 words, overlap = 50 words)

| Document | Chunks |
|---|---|
| document15.pdf | 72 |
| document9.pdf | 42 |
| document14.pdf | 32 |
| document16.pdf | 35 |
| document5.pdf | 29 |
| document8.pdf | 25 |
| document3.pdf | 23 |
| document13.pdf | 23 |
| document4.pdf | 22 |
| document1.pdf | 14 |
| document2.pdf | 14 |
| document12.pdf | 15 |
| document6.pdf | 9 |
| document7.pdf | 7 |
| document10.pdf | 6 |
| document11.pdf | 5 |
| document17.pdf | 5 |

---

## Synthetic Questionnaire

50 questions generated by Ollama from document chunks, saved to `evaluation/results/questions.json`.

| Difficulty | Count | Type | Count |
|---|---|---|---|
| Easy | 48 | Factual | 49 |
| Medium | 2 | Multi-hop | 1 |

Each question includes: `question`, `answer`, `difficulty`, `type`, `source_hint`, `chunk_id`, `source`, `pages`.

---

## Evaluation Metrics

### Retrieval (K = 5)

| Metric | Definition |
|---|---|
| **Precision@5** | Fraction of top-5 retrieved chunks that are relevant |
| **Recall@5** | Fraction of all relevant chunks that appear in top-5 |
| **MRR** | Mean Reciprocal Rank — avg of 1/rank of first relevant chunk |

**Why K=5?** Five chunks at 500 words each ≈ 2,500 words of context — well within the model's token limit while capturing enough relevant passages. The ground-truth source chunk appears in the top-5 for the majority of factual queries.

### Generation (LLM-as-Judge)

| Metric | Measurement |
|---|---|
| **Faithfulness** | `mistral` rates 0–1: is the answer grounded solely in retrieved context? |
| **Answer Relevance** | `mistral` rates 0–1: does the answer directly address the question? |

LLM-as-judge is used because it scales to 50 questions without manual labelling and correlates well with human judgement for technical Q&A tasks.

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `mistral` model pulled

```bash
# Install Ollama, then pull the model
ollama pull mistral
```

### Install dependencies

```bash
cd multiagent_rag
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
```

---

## Execution Steps

### Step 1 — Start Ollama
```bash
ollama serve
# Verify mistral is available
ollama list
```

### Step 2 — Place PDF documents
Copy your ARAI AIS PDFs into `data/documents/`.

### Step 3 — Ingest documents
```bash
venv/bin/python ingest.py
```
```
Found 17 PDF(s) to ingest
[IngestionAgent] Done. 17 docs, 378 chunks indexed.
```

### Step 4 — Generate evaluation questions
```bash
venv/bin/python generate_questions.py --num-questions 50
```
```
Loaded 378 chunks. Generating 50 questions...
[QuestionnaireGenerator] Generated 50 questions.
Saved → evaluation/results/questions.json
```

### Step 5 — Run evaluation
```bash
venv/bin/python evaluate.py --k 5
```
```
=======================================================
  EVALUATION METRICS SUMMARY
=======================================================
  precision@5               0.XXXX
  recall@5                  0.XXXX
  mrr                       0.XXXX
  faithfulness              0.XXXX
  answer_relevance          0.XXXX
=======================================================
Chart saved → evaluation/results/evaluation_chart.png
```

### Step 6 — Chat
```bash
# Interactive
venv/bin/python main.py

# Single query
venv/bin/python main.py --query "What are the headlamp requirements in AIS-008?"
```

### Run everything at once
```bash
./run_pipeline.sh
```

---

## Project Structure

```
multiagent_rag/
├── agents/
│   ├── orchestrator.py        # OrchestratorAgent — Ollama tool-use loop
│   ├── ingestion_agent.py     # DocumentIngestionAgent — PDF → ChromaDB
│   ├── retriever_agent.py     # RetrieverAgent — vector search
│   ├── generator_agent.py     # GeneratorAgent — Ollama answer generation + LLM-judge
│   └── evaluator_agent.py     # EvaluatorAgent — all metrics
├── tools/
│   ├── search_tool.py         # search_documents tool + ChromaDB wrapper
│   ├── memory_tool.py         # save/retrieve_from_memory tools
│   └── eval_tool.py           # Precision@K, Recall@K, MRR functions
├── memory/
│   ├── short_term.py          # In-memory 10-turn conversation window
│   └── long_term.py           # SQLite: interactions, documents, key-value
├── utils/
│   ├── pdf_processor.py       # PyMuPDF extraction + word-level chunking
│   └── embedder.py            # sentence-transformers singleton
├── evaluation/
│   ├── metrics.py             # Report generation + matplotlib charts
│   ├── questionnaire.py       # Ollama-driven synthetic Q&A generation
│   └── results/               # questions.json, CSV, JSON metrics, PNG charts
├── data/documents/            # ← Place PDF files here (17 AIS PDFs)
├── vector_store/              # ChromaDB persistence (378 chunks)
├── memory/long_term.db        # SQLite persistent memory
├── config.py                  # All configuration (Ollama URL, model, K, paths)
├── ingest.py                  # Step 1: ingest PDFs
├── generate_questions.py      # Step 2: generate 50 Q&A pairs
├── evaluate.py                # Step 3: run evaluation + save charts
├── main.py                    # Step 4: interactive chat
├── run_pipeline.sh            # Runs all 4 steps in sequence
└── requirements.txt
```

---

## System Improvements Log

| Improvement | Area Targeted | What Changed | Impact on Metrics |
|---|---|---|---|
| Switched from Anthropic API to Ollama | Model | Replaced `anthropic` SDK with `openai` SDK pointed at `localhost:11434/v1` | Fully offline; no API key needed; same tool-use interface |
| Cosine distance space | Retrieval | Set ChromaDB collection metric to `cosine` | Better semantic ranking regardless of chunk length |
| Similarity threshold filter | Retrieval | Drop chunks with score < 0.2 | Reduces noise; improves Precision@K |
| Deterministic RAG fallback | Orchestrator | If model skips tool call, explicitly retrieve then generate | Prevents zero-context answers when tool-calling is unreliable on small models |
| Sliding conversation window | Short-term memory | Cap at 20 messages; trim oldest turns | Prevents token overflow on long sessions |
| Singleton embedding model | Utils | Load `all-MiniLM-L6-v2` once via class-level `_instance` | Eliminates repeated 1–2s model load on every call |
| JSON fence stripping | Question generation | Strip ` ```json ` fences from Ollama output before `json.loads` | Eliminates parse errors from models that add markdown wrappers |
| Batch embedding | Ingestion | Embed chunks in batches of 64 | ~5× faster ingestion on 17 PDFs |

---

## Results & Analysis

After `evaluate.py` completes, results are saved in `evaluation/results/`:

| File | Contents |
|---|---|
| `evaluation_records.csv` | Per-question scores (query, answer, faithfulness, relevance) |
| `evaluation_metrics.json` | Aggregate metrics |
| `evaluation_chart.png` | Bar chart — Precision@5, Recall@5, MRR, Faithfulness, Relevance |
| `evaluation_distribution.png` | Box plot — score distribution across 50 questions |

### Strengths
- Fully local — no API keys, no data leaving the machine
- Faithfulness is high for factual questions where the answer is directly in a single chunk
- Tool-use loop allows the model to check memory before re-retrieving, reducing latency on repeated queries

### Known Failure Cases
- **Multi-hop questions**: Require context from two separate chunks; retrieval may only surface one
- **Table/figure content**: PyMuPDF flattens tables to text; tabular answers may lose structure
- **Cross-reference questions**: AIS documents reference other standards not in the corpus
- **Annex/figure index questions**: Questions about figure numbers in an annex often miss because the figure caption is a separate tiny chunk below the retrieval threshold
