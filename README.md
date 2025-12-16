# VNPT AI Water Margin

Production-ready runner for VNPT-hosted LLMs to answer multiple-choice questions with advanced Retrieval-Augmented Generation (RAG), progress persistence, and multi-format document support.

## ✨ Key Features

- 🎯 **Domain-Based Routing** - Intelligent model selection based on question type (NEW!)
  - SAFETY_REFUSAL → Small model (cost-effective)
  - NON_RAG → Large model with Chain-of-Thought reasoning
  - RAG_NECESSITY → Large model with knowledge retrieval
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

**New in v2.0**: Enable intelligent question routing:
```dotenv
# Domain-Based Routing (automatically uses small/large model based on question type)
DOMAIN_ROUTING_ENABLED=true  # Default: enabled
```

### 4. Run

```pwsh
uv run main.py
```

## 🎯 Domain-Based Routing (NEW!)

Automatically routes questions to optimal models based on classification:

| Domain | Model | Temperature | RAG | Use Case |
|--------|-------|-------------|-----|----------|
| **SAFETY_REFUSAL** | Small | 0.3 | No | Ethical/legal violations |
| **NON_RAG** | Large | 0.7 | No | Math, code, reading comprehension |
| **RAG_NECESSITY** | Large | 0.5 | Yes | Knowledge-based questions |

### How It Works

1. **Classify questions** (if not already done):
   ```python
   from src.classification.classify import process_classification_dataset
   
   process_classification_dataset(
       input_file='data/test.json',
       output_file='data/test_classification.json',
       config=config
   )
   ```

2. **Run with routing** (automatic):
   ```pwsh
   # Loads data/test_classification.json
   # Merges with data/test.json for full questions
   # Routes each question to appropriate model
   uv run main.py
   ```

3. **Customize routing** (optional):
   - Edit `src/utils/prompts_config.py` for custom prompts
   - Modify LLM parameters per domain
   - Change model selection rules

### Disable Routing

To use traditional single-model approach:
```dotenv
# In .env
DOMAIN_ROUTING_ENABLED=false
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

### Domain-Based Routing (NEW!)

Control intelligent question routing:

```dotenv
# Enable/disable smart routing
DOMAIN_ROUTING_ENABLED=true

# Routing automatically selects:
# - Small model for SAFETY_REFUSAL (temperature=0.3)
# - Large model + CoT for NON_RAG (temperature=0.7)
# - Large model + RAG for RAG_NECESSITY (temperature=0.5)
```

**Customize routing**: Edit `src/utils/prompts_config.py`

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

The codebase is organized into **modular packages** for maintainability and clarity:

```
VNPT_AI_Water_Margin/
├── .secret/
│   └── api-keys.json           # Auto-loaded credentials
├── data/
│   └── test.json               # Input dataset (~370 questions)
├── docs/                       # Knowledge base documents
│   └── *.pdf, *.md, *.csv      # Multi-format support
├── knowledge_base/             # Generated indices
│   ├── faiss_index.bin         # Dense semantic search
│   ├── bm25_index.pkl          # Sparse keyword search
│   └── text_chunks.json        # Processed text chunks
├── results/                    # Output with progress
│   ├── test_vnpt_async.csv     # Main results
│   └── submission.csv          # Competition format
├── src/
│   ├── utils/                  # 🔧 Utility modules
│   │   ├── prompt.py           # Prompt formatting
│   │   ├── prediction.py       # Answer extraction
│   │   └── progress.py         # Checkpoint/resume
│   ├── RAG/                    # 🔍 Retrieval-Augmented Generation
│   │   ├── build_index.py      # Index builder (run once)
│   │   ├── loader.py           # Load indices & models
│   │   ├── retriever.py        # Hybrid search & re-ranking
│   │   └── pre_retrieve.py     # Pre-compute context
│   ├── core/                   # ⚙️ Core processing
│   │   ├── config.py           # Configuration defaults
│   │   ├── processor.py        # Single item processing
│   │   └── runner.py           # Async orchestration
│   ├── providers/              # 🔌 Provider implementations
│   │   ├── vnpt.py             # VNPT AI provider
│   │   ├── ollama.py           # Ollama local LLMs
│   │   ├── openai.py           # OpenAI API
│   │   └── huggingface.py      # HuggingFace embeddings
│   ├── logger.py               # Centralized logging
│   └── async_running.py        # Backwards compat wrapper
├── main.py                     # 🚀 Entry point
└── .env                        # Configuration
```

## 🔄 Code Workflow

### Overview

The application follows a **pipeline architecture** with clear separation of concerns:

```
main.py 
   ↓
core/runner.py ──→ Choose Mode (Test/Validation)
   ↓
RAG/loader.py ──→ Load indices (FAISS, BM25, CrossEncoder)
   ↓
utils/progress.py ──→ Load checkpoint & filter processed items
   ↓
core/processor.py ──→ Process items in parallel ⚡
   ├─→ RAG/retriever.py ──→ Retrieve context (if enabled)
   ├─→ utils/prompt.py ──→ Format question into messages
   ├─→ providers/vnpt.py ──→ Call LLM API
   └─→ utils/prediction.py ──→ Extract answer (A, B, C, D)
   ↓
Save to CSV ──→ results/test_vnpt_async.csv
```

### Detailed Module Interactions

#### 1. Entry Point (`main.py`)
```python
# Loads config from .env
config = _build_config_from_env()

# Calls the main orchestrator
asyncio.run(process_dataset_async(
    input_file='data/test.json',
    output_file='results/test_vnpt_async.csv',
    config=config,
    mode='test'
))
```

#### 2. Orchestration (`core/runner.py`)
- **`process_dataset()`** - Routes to test or validation mode
- **`run_test_mode()`** - Processes questions without ground truth
  - ✅ Supports checkpoint/resume
  - ✅ Generates submission file
- **`run_validation_mode()`** - Processes with accuracy calculation
  - ✅ Shows real-time accuracy
  - ✅ Saves detailed results

#### 3. RAG Pipeline (`RAG/loader.py` + `RAG/retriever.py`)

**Loading Phase:**
```python
# RAG/loader.py - load_rag_components()
faiss_index = faiss.read_index("knowledge_base/faiss_index.bin")
bm25_index = pickle.load("knowledge_base/bm25_index.pkl")
text_chunks = json.load("knowledge_base/text_chunks.json")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

**Retrieval Phase:**
```python
# RAG/retriever.py - retrieve_context()
if hybrid_search_enabled:
    # Combine FAISS (semantic) + BM25 (keyword)
    chunks = _hybrid_search(question, ...)
else:
    # FAISS only (semantic)
    chunks = _dense_search(question, ...)

if rerank_enabled:
    # Re-rank with CrossEncoder
    final_chunks = _rerank_chunks(question, chunks, ...)

return "\n".join(final_chunks)
```

#### 4. Item Processing (`core/processor.py`)

For each question item:
```python
async def process_item(item, provider, config, ...):
    # 1. Retrieve context (if RAG enabled)
    context = await retrieve_context(question, ...)
    
    # 2. Format prompt
    messages = format_prompt(item, context)
    
    # 3. Call LLM
    prediction_text = await provider.achat(messages, config)
    
    # 4. Extract answer
    answer = clean_prediction(prediction_text)
    
    # 5. Return result
    return {"qid": ..., "answer": answer, ...}
```

#### 5. Utilities

**Prompt Formatting (`utils/prompt.py`):**
```python
format_prompt(item, context=None)
# → [{"role": "system", "content": "..."}, 
#    {"role": "user", "content": "Question: ..."}]
```

**Answer Extraction (`utils/prediction.py`):**
```python
clean_prediction("Đáp án: B. Vì...")
# → "B"
```

**Progress Management (`utils/progress.py`):**
```python
processed_qids, results = load_progress("results/test.csv")
items_to_process = filter_items(data, processed_qids)
# → Only process remaining items
```

### Parallel Execution Flow

The system processes multiple items concurrently:

```
Question 1 ──→ [Retrieve] ──→ [Prompt] ──→ [LLM] ──→ [Parse] ──→ CSV
Question 2 ──→ [Retrieve] ──→ [Prompt] ──→ [LLM] ──→ [Parse] ──→ CSV
Question 3 ──→ [Retrieve] ──→ [Prompt] ──→ [LLM] ──→ [Parse] ──→ CSV
    ⋮                (controlled by CONCURRENT_REQUESTS=2)
```

### Data Flow Example

```
1. Input: data/test.json
   {"qid": "001", "question": "What is AI?", "choices": ["A", "B", "C", "D"]}

2. RAG Retrieval (if enabled):
   "Context: AI stands for Artificial Intelligence..."

3. Prompt:
   "Context: ...\nQuestion: What is AI?\nA. Robot\nB. Intelligence\n..."

4. LLM Response:
   "Phân tích: AI là... Đáp án: B"

5. Parsed Answer:
   "B"

6. Output: results/test_vnpt_async.csv
   qid,answer,prediction_raw
   001,B,"Phân tích: AI là... Đáp án: B"
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

- **[CHANGELOG.md](CHANGELOG.md)**: Version history and migration guide
- **[docs/domain_routing_config.md](docs/domain_routing_config.md)**: Domain routing configuration guide
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Code architecture & workflow guide with diagrams
- **[AGENTS.md](AGENTS.md)**: Comprehensive agent/architecture overview
- **[docs/credentials.md](docs/credentials.md)**: Credential management details
- **[docs/infinite_retry.md](docs/infinite_retry.md)**: Retry mechanism documentation
- **[docs/quick-start-json-credentials.md](docs/quick-start-json-credentials.md)**: JSON credential setup guide