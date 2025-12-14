# VNPT AI Water Margin

Production-ready runner for VNPT-hosted LLMs to answer multiple-choice questions with advanced Retrieval-Augmented Generation (RAG), progress persistence, and multi-format document support.

## ✨ Key Features

- 🚀 **Progress Resume** - Automatic checkpoint/resume on interruption
- 📁 **Multi-Format Support** - PDF, JSON, CSV, XLSX, DOCX, MD, TXT
- 🧠 **Smart Chunking** - LangChain-powered semantic chunking with special handling for markdown and tabular data
- 🔄 **Infinite Retry** - Automatic quota limit handling with exponential backoff
- 🔐 **JSON Credentials** - Zero-configuration with `.secret/api-keys.json`
- ⚙️ **Fully Configurable** - All parameters via `.env` file

## Quick Start

### 1. Installation

```pwsh
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
uv pip install -r requirements.txt
```

### 2. Credential Setup (Zero Config!)

Create `.secret/api-keys.json`:

```json
[
  {
    "authorization": "Bearer YOUR_EMBEDDING_TOKEN",
    "tokenKey": "YOUR_EMBEDDING_TOKEN_KEY",
    "llmApiName": "LLM embeddings",
    "tokenId": "YOUR_EMBEDDING_TOKEN_ID"
  },
  {
    "authorization": "Bearer YOUR_LARGE_MODEL_TOKEN",
    "tokenKey": "YOUR_LARGE_TOKEN_KEY",
    "llmApiName": "LLM large",
    "tokenId": "YOUR_LARGE_TOKEN_ID"
  },
  {
    "authorization": "Bearer YOUR_SMALL_MODEL_TOKEN",
    "tokenKey": "YOUR_SMALL_TOKEN_KEY",
    "llmApiName": "LLM small",
    "tokenId": "YOUR_SMALL_TOKEN_ID"
  }
]
```

### 3. Configure Environment

Copy `.env.example` to `.env` (credentials auto-loaded from JSON):

```pwsh
copy .env.example .env
```

### 4. Run

```pwsh
uv run main.py
```

## 🔄 Progress Resume

Progress is automatically saved! If interrupted (Ctrl+C), just run again:

```
📋 Found existing progress file: results/test_vnpt_async.csv
✅ Loaded 31 already processed items
📊 Progress: 31/370 complete. Processing 340 remaining items...
```

## 📚 Retrieval-Augmented Generation (RAG)

### Supported Document Formats

Place any of these in your `docs/` folder:

- **PDF** (`.pdf`) - Research papers, books
- **JSON** (`.json`) - Structured data with smart chunking
- **CSV** (`.csv`) - Tabular data with header preservation
- **Excel** (`.xlsx`, `.xls`) - Spreadsheets (requires `pandas`)
- **Word** (`.docx`, `.doc`) - Documents (requires `python-docx`)
- **Markdown** (`.md`) - Documentation with structure-aware chunking
- **Text** (`.txt`) - Plain text files

### Build Knowledge Base

```pwsh
uv run .\\src\\RAG\\build_index.py
```

**Advanced Chunking:**
- **Markdown files**: Preserves headers, lists, code blocks
- **Tabular data** (JSON/CSV/XLSX): Preserves row integrity
- **Text files**: LangChain's RecursiveCharacterTextSplitter for semantic coherence

**Configuration (.env):**
```dotenv
RAG_ENABLED=true
RETRIEVE_DOCS_DIR=docs           # Custom docs directory
RAG_CHUNK_SIZE=500               # Chunk size in characters
RAG_CHUNK_OVERLAP=50             # Overlap between chunks
EMBEDDING_DIM=768                # Embedding dimension
EMBEDDING_MODEL_NAME=vnptai_hackathon_embedding
```

### Advanced RAG: Hybrid Search + Re-ranking

**Hybrid Search** (combines semantic + keyword):
```dotenv
HYBRID_SEARCH_ENABLED=true
SEMANTIC_WEIGHT=0.5              # FAISS weight
KEYWORD_WEIGHT=0.5               # BM25 weight
```

**Re-ranking** (improves precision):
```dotenv
RERANK_ENABLED=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10                  # Initial retrieval count
TOP_K_RAG=3                      # Final chunks for LLM
```

### Pre-retrieve Context (Optional)

For repeated runs, pre-compute context:

```pwsh
uv run .\\src\\RAG\\pre_retrieve.py
```

Then enable in `.env`:
```dotenv
USE_PRE_RETRIEVED_CONTEXT=true
```

##  Configuration

### LLM Hyperparameters

All configurable via `.env`:

```dotenv
LLM_TEMPERATURE=0.5              # Randomness (0.0-1.0)
LLM_TOP_P=0.7                    # Nucleus sampling
LLM_MAX_TOKENS=2048              # Max completion tokens
LLM_N=1                          # Number of completions
LLM_SEED=416                     # Reproducibility seed
```

### Performance & Rate Limiting

```dotenv
CONCURRENT_REQUESTS=2            # Parallel requests
SLEEP_TIME=0                     # Let retry handle delays
```

### Infinite Retry (VNPT)

Automatic handling of quota limits:

```dotenv
VNPT_INFINITE_RETRY=true         # Enable infinite retry
VNPT_RETRY_INITIAL_DELAY=5       # Initial delay (seconds)
VNPT_RETRY_MAX_DELAY=300         # Max delay (5 minutes)
```

### Credential Priority

1. **Model-specific env vars** (highest): `VNPT_LARGE_ACCESS_TOKEN`
2. **`.secret/api-keys.json`** (recommended, default)
3. **Generic env vars** (fallback): `VNPT_ACCESS_TOKEN`

## Multiple Providers

### VNPT (Default)

```dotenv
CHAT_PROVIDER=vnpt
MODEL_NAME=vnptai-hackathon-large
# Credentials from .secret/api-keys.json
```

### Ollama (Local)

```dotenv
CHAT_PROVIDER=ollama
MODEL_NAME=gemma2:270m
OLLAMA_BASE=http://localhost:11434

EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### OpenAI

```dotenv
CHAT_PROVIDER=openai
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_key_here
```

## 📊 Logging

Structured logging to console and `logs/app.log`:

```dotenv
LOG_LEVEL=INFO                   # INFO, DEBUG, WARNING, ERROR
```


## 📁 Project Structure

```
VNPT_AI_Water_Margin/
├── .secret/
│   └── api-keys.json           # Auto-loaded credentials
├── data/
│   └── test.json               # Input dataset
├── docs/                       # Knowledge base documents
├── knowledge_base/             # Generated indices
│   ├── faiss_index.bin
│   ├── bm25_index.pkl
│   └── text_chunks.json
├── results/                    # Output with progress
│   └── test_vnpt_async.csv
├── src/
│   ├── providers/              # Provider implementations
│   ├── RAG/                    # RAG pipeline
│   │   ├── build_index.py     # Index builder
│   │   └── pre_retrieve.py    # Pre-retrieval script
│   └── async_running.py        # Main processing engine
├── main.py                     # Entry point
└── .env                        # Configuration
```

## 🔧 Advanced Features

### Custom Providers

Implement in `src/providers/`. Chat providers need `achat()`, embedding providers need `aembed()`.

### Content Filtering

RAG automatically filters irrelevant content (API docs, code snippets, etc.) from retrieved context.

### Tabular Data Intelligence

JSON/CSV/XLSX files are chunked to preserve:
- Row integrity
- Header context
- Table structure

### Markdown Structure

Markdown files maintain:
- Header hierarchy
- List formatting
- Code block boundaries

## 📝 Notes

- **Secrets**: Never commit `.secret/api-keys.json` or `.env` with credentials
- **Progress**: Results in `results/` folder with automatic resume
- **Rate Limits**: Infinite retry handles VNPT quotas automatically
- **Extensibility**: Easy to add new providers and document formats

## 📖 Additional Documentation

- **[AGENTS.md](AGENTS.md)**: Comprehensive agent/architecture overview
- **[docs/credentials.md](docs/credentials.md)**: Credential management details
- **[docs/infinite_retry.md](docs/infinite_retry.md)**: Retry mechanism documentation
- **[docs/quick-start-json-credentials.md](docs/quick-start-json-credentials.md)**: JSON credential setup guide