# Agents Overview: VNPT AI Water Margin Project

This project provides a robust and extensible framework for answering multiple-choice questions by leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Its modular architecture allows for flexible integration with various LLM and embedding providers, alongside advanced retrieval strategies.

## Overall Project Architecture

The core of the project revolves around an asynchronous runner (`main.py` / `src/async_running.py`) that processes datasets of multiple-choice questions. Key architectural components include:

1.  **Modular Provider System:** Decoupled interfaces for Chat Completion and Embedding generation, allowing different backend services to be used independently.
2.  **Asynchronous Processing:** Utilizes `asyncio` for concurrent execution of LLM requests, significantly improving throughput while respecting API rate limits.
3.  **Advanced RAG Pipeline:** Integrates sophisticated document retrieval and augmentation techniques to provide LLMs with relevant context, thereby enhancing answer accuracy.
4.  **Progress Persistence:** Automatic checkpoint/resume functionality saves progress and allows resuming interrupted runs without data loss.
5.  **Logging:** A centralized logging system (`src/logger.py`) ensures that all critical operations, errors, and debugging information are recorded for future analysis.

## Credential Management

### JSON-Based Credentials (Recommended)

The project prioritizes credentials from `.secret/api-keys.json` for a zero-configuration experience:

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

**Credential Loading Priority:**
1. Model-specific environment variables (e.g., `VNPT_LARGE_ACCESS_TOKEN`)
2. `.secret/api-keys.json` (default, automatic)
3. Generic environment variables (e.g., `VNPT_ACCESS_TOKEN`)

This allows seamless operation with just the JSON file while supporting environment variable overrides when needed.

## RAG Pipeline Details

The RAG pipeline efficiently retrieves and synthesizes relevant information from a knowledge base to augment LLM prompts.

### 1. Knowledge Base Construction (`src/RAG/build_index.py`)

**Multi-Format Document Support:**
- **PDF** (`.pdf`) - Text extraction via pypdf
- **JSON** (`.json`) - Structured data with smart chunking
- **CSV** (`.csv`) - Tabular data with header preservation
- **Excel** (`.xlsx`, `.xls`) - Spreadsheet data via pandas
- **Word** (`.docx`, `.doc`) - Document text via python-docx
- **Markdown** (`.md`) - Structure-aware chunking
- **Text** (`.txt`) - Plain text files

**Advanced Chunking Strategies:**
- **LangChain Integration:** Uses `RecursiveCharacterTextSplitter` for semantic coherent chunks
- **Markdown-Aware:** Special `MarkdownTextSplitter` preserves document structure (headers, lists, code blocks)
- **Tabular Data Handling:** Specialized chunking for JSON/CSV/XLSX that preserves row integrity and table structure
- **Configurable Parameters:** `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `EMBEDDING_DIM` via `.env`

**Dual Indexing:**
- **Dense Index (FAISS):** Embeddings for semantic similarity search
- **Sparse Index (BM25):** Keyword-based index for lexical relevance

**Configurable Document Source:**
- Set `RETRIEVE_DOCS_DIR` in `.env` to specify custom document directory

### 2. Context Retrieval (`src/async_running.py`)

**Hybrid Search:** Combines FAISS and BM25 searches with configurable weights (`SEMANTIC_WEIGHT`, `KEYWORD_WEIGHT`)

**Re-ranking:** CrossEncoder model provides fine-grained relevance evaluation

**Content Filtering:** Automatically filters out irrelevant technical content (API docs, code snippets, etc.)

**Prompt Augmentation:** Most relevant chunks prepended to questions

### 3. Pre-retrieval Optimization (`src/RAG/pre_retrieve.py`)

Offline retrieval execution for repeated evaluations. When `USE_PRE_RETRIEVED_CONTEXT=true`, uses pre-computed context, skipping real-time retrieval.

## Chat Providers

### 1. VNPT AI (Primary)

**Configuration:**
- Model-specific credentials automatically loaded from `.secret/api-keys.json`
- Supports `vnptai-hackathon-small` and `vnptai-hackathon-large`
- Separate embedding model credentials (`vnptai_hackathon_embedding`)

**Features:**
- **Infinite Retry:** Automatic retry with exponential backoff for quota/rate limits
- **Configurable via `.env`:**
  - `VNPT_INFINITE_RETRY=true` (default)
  - `VNPT_RETRY_INITIAL_DELAY=5` (seconds)
  - `VNPT_RETRY_MAX_DELAY=300` (seconds)

**Endpoints:**
- Chat: `/data-service/v1/chat/completions/{model_name}`
- Embedding: `/data-service/vnptai-hackathon-embedding`

### 2. Ollama

Supports local/remote Ollama instances with open-source LLMs.

**Configuration:** `CHAT_PROVIDER=ollama`, `OLLAMA_BASE=http://localhost:11434`

### 3. OpenAI

Integrates with GPT-3.5, GPT-4, etc.

**Configuration:** `CHAT_PROVIDER=openai`, `OPENAI_API_KEY=...`

## Embedding Providers

### 1. VNPT AI

Uses VNPT AI embedding model (`vnptai_hackathon_embedding`). Credentials loaded from `.secret/api-keys.json` (embedding section).

### 2. Hugging Face

Local `sentence-transformers` models for offline development.

**Configuration:** `EMBEDDING_PROVIDER=huggingface`, `HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2`

## Key Configuration Variables

### Core Settings
- **`CHAT_PROVIDER`**: LLM provider (`vnpt`, `ollama`, `openai`)
- **`EMBEDDING_PROVIDER`**: Embedding provider (`vnpt`, `huggingface`)
- **`MODEL_NAME`**: Chat model name (e.g., `vnptai-hackathon-large`)
- **`EMBEDDING_MODEL_NAME`**: Embedding model name (default: `vnptai_hackathon_embedding`)

### Performance & Rate Limiting
- **`CONCURRENT_REQUESTS`**: Simultaneous LLM requests (default: 2)
- **`SLEEP_TIME`**: Delay between requests (default: 0, retry handles delays)

### LLM Hyperparameters
- **`LLM_TEMPERATURE`**: Randomness (0.0-1.0, default: 0.5)
- **`LLM_TOP_P`**: Nucleus sampling (0.0-1.0, default: 0.7)
- **`LLM_MAX_TOKENS`**: Max completion tokens (default: 2048)
- **`LLM_N`**: Number of completions (default: 1)
- **`LLM_SEED`**: Random seed for reproducibility (default: 416)

### RAG Configuration
- **`RAG_ENABLED`**: Enable/disable RAG pipeline
- **`RETRIEVE_DOCS_DIR`**: Document source directory (default: `docs`)
- **`RAG_CHUNK_SIZE`**: Chunk size in characters (default: 500)
- **`RAG_CHUNK_OVERLAP`**: Overlap between chunks (default: 50)
- **`EMBEDDING_DIM`**: Embedding dimension (default: 768)
- **`TOP_K_RAG`**: Number of chunks to retrieve (default: 3)
- **`FAISS_INDEX_PATH`**: FAISS index file path
- **`TEXT_CHUNKS_PATH`**: Text chunks JSON path
- **`BM25_INDEX_PATH`**: BM25 index file path

### Advanced RAG
- **`HYBRID_SEARCH_ENABLED`**: Enable hybrid retrieval
- **`SEMANTIC_WEIGHT`**: FAISS search weight (0.0-1.0, default: 0.5)
- **`KEYWORD_WEIGHT`**: BM25 search weight (0.0-1.0, default: 0.5)
- **`RERANK_ENABLED`**: Enable re-ranking
- **`CROSS_ENCODER_MODEL`**: Re-ranking model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **`RERANK_TOP_K`**: Initial retrieval count for re-ranking (default: 10)
- **`USE_PRE_RETRIEVED_CONTEXT`**: Use pre-computed context

### VNPT Retry Configuration
- **`VNPT_INFINITE_RETRY`**: Enable infinite retry (default: true)
- **`VNPT_RETRY_INITIAL_DELAY`**: Initial delay in seconds (default: 5)
- **`VNPT_RETRY_MAX_DELAY`**: Maximum delay in seconds (default: 300)

### System
- **`LOG_LEVEL`**: Logging verbosity (`INFO`, `DEBUG`, etc.)

## Progress Persistence

The system automatically saves progress to CSV files and resumes from the last checkpoint:

- **Checkpoint Creation:** Results written incrementally to CSV
- **Resume Capability:** On restart, skips already-processed items
- **Status Display:** Shows progress (`31/370 complete. Processing 340 remaining...`)

No configuration needed - works automatically!

This comprehensive overview provides complete understanding of the project's capabilities, configuration options, and advanced features.
