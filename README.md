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

### Quick Submission

```bash
# 1. Build Docker image
docker build -t your_username/vnpt-ai-water-margin:latest .

# 2. Test locally
docker run --gpus all \
  -v $(pwd)/data/test.json:/code/private_test.json \
  your_username/vnpt-ai-water-margin:latest

# 3. Push to DockerHub
docker push your_username/vnpt-ai-water-margin:latest
```

### What Gets Submitted
- **GitHub Repository URL** (public)
- **DockerHub Image Name**: `your_username/vnpt-ai-water-margin:latest`

### Architecture for Submission

The Docker container runs:
1. **`process_data.py`** - Initialize vector database from `docs/`
2. **`inference.py`** - Classify + answer questions in parallel
3. **Outputs**: `submission.csv` (qid, answer) + `submission_time.csv` (qid, answer, time)

**Key Features for Competition:**
- ✅ Parallel processing (classify + answer per question)
- ✅ CUDA 12.2 support
- ✅ Automatic RAG initialization
- ✅ Ordered output by QID
- ✅ Timing per question

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
│   │   ├── prompts_config.py   # Domain-based routing config
│   │   └── progress.py         # Checkpoint/resume
│   ├── rag/                    # 🔍 Retrieval-Augmented Generation
│   │   ├── build_index.py      # Index builder (run once)
│   │   ├── loader.py           # Load indices & models
│   │   ├── retriever.py        # Hybrid search & re-ranking
│   │   └── pre_retrieve.py     # Pre-compute context
│   ├── core/                   # ⚙️ Core processing
│   │   ├── config.py           # Configuration defaults
│   │   ├── processor.py        # Single item processing
│   │   └── runner.py           # Async orchestration
│   ├── classification/         # 🏷️ Question classification
│   │   └── classify.py         # Domain classifier
│   ├── providers/              # 🔌 Provider implementations
│   │   ├── vnpt.py             # VNPT AI provider
│   │   ├── ollama.py           # Ollama local LLMs
│   │   ├── openai.py           # OpenAI API
│   │   └── huggingface.py      # HuggingFace embeddings
│   ├── logger.py               # Centralized logging
│   └── async_running.py        # Backwards compat wrapper
├── process_data.py             # 🔧 Vector DB initialization (Docker)
├── inference.py                # 🎯 Main inference pipeline (Docker)
├── inference.sh                # 🐚 Shell entry point (Docker)
├── main.py                     # 🚀 Development entry point
├── Dockerfile                  # 🐳 Container definition
└── .env                        # Configuration
```

---

## 🔄 Pipeline Flow (Competition Submission)

### System Architecture Diagram

```mermaid
flowchart TD
    A[Input: /code/private_test.json] --> B[Load Questions]
    B --> C[Initialize Components]
    
    C --> D[Load Chat Provider]
    C --> E[Load RAG Components]
    
    E --> E1[FAISS Index]
    E --> E2[BM25 Index]
    E --> E3[Cross-Encoder]
    
    D --> F[Create Parallel Tasks]
    E1 --> F
    E2 --> F
    E3 --> F
    
    F --> G[Process Questions in Parallel]
    
    G --> H1[Question 1]
    G --> H2[Question 2]
    G --> H3[Question N...]
    
    H1 --> I1[Classify Domain]
    H2 --> I2[Classify Domain]
    H3 --> I3[Classify Domain]
    
    I1 --> J1{Domain?}
    I2 --> J2{Domain?}
    I3 --> J3{Domain?}
    
    J1 --> |SAFETY_REFUSAL| K1[Large Model<br/>Temp=0.2<br/>No RAG]
    J1 --> |NON_RAG| K2[Small Model<br/>Temp=0.3<br/>With RAG]
    J1 --> |RAG_NECESSITY| K3[Small Model<br/>Temp=0.2<br/>With RAG]
    
    J2 --> |Route| K4[Selected Model]
    J3 --> |Route| K5[Selected Model]
    
    K2 --> L1[Retrieve Context]
    K3 --> L2[Retrieve Context]
    
    L1 --> M1[Format Prompt]
    L2 --> M2[Format Prompt]
    K1 --> M3[Format Prompt]
    K4 --> M4[Format Prompt]
    K5 --> M5[Format Prompt]
    
    M1 --> N1[Call LLM]
    M2 --> N2[Call LLM]
    M3 --> N3[Call LLM]
    M4 --> N4[Call LLM]
    M5 --> N5[Call LLM]
    
    N1 --> O1[Parse Answer + Time]
    N2 --> O2[Parse Answer + Time]
    N3 --> O3[Parse Answer + Time]
    N4 --> O4[Parse Answer + Time]
    N5 --> O5[Parse Answer + Time]
    
    O1 --> P[Collect All Results]
    O2 --> P
    O3 --> P
    O4 --> P
    O5 --> P
    
    P --> Q[Sort by QID]
    Q --> R1[submission.csv]
    Q --> R2[submission_time.csv]
```

### Execution Flow

```mermaid
sequenceDiagram
    participant D as Docker Container
    participant P as process_data.py
    participant I as inference.py
    participant C as Classifier
    participant R as RAG System
    participant L as LLM Provider
    participant O as Output Files
    
    D->>P: 1. Run initialization
    P->>P: Check if indices exist
    alt Indices not exist
        P->>R: Build FAISS + BM25 indices
        R->>P: Indices ready
    else Indices exist
        P->>P: Skip (already initialized)
    end
    
    D->>I: 2. Run inference
    I->>I: Load test data
    I->>I: Create parallel tasks
    
    par For each question (parallel)
        I->>C: Classify question
        C->>I: Domain label
        
        alt Domain requires RAG
            I->>R: Retrieve context
            R->>I: Relevant chunks
        end
        
        I->>L: Call LLM with domain config
        L->>I: Raw prediction
        I->>I: Parse answer + record time
    end
    
    I->>I: Sort results by QID
    I->>O: Write submission.csv
    I->>O: Write submission_time.csv
    O->>D: Outputs ready
```

---

## 📊 Data Processing Steps

### 1. Input Data Format

**Expected Input:** JSON file with questions
```json
[
  {
    "qid": "q001",
    "question": "Question text here?",
    "choices": ["Option A", "Option B", "Option C", "Option D"]
  },
  ...
]
```

**Location (Docker):** `/code/private_test.json`

### 2. Classification Stage

**Purpose:** Determine optimal processing strategy for each question

**Process:**
1. **Input:** Question text + choices
2. **Classification Prompt:** Analyzes question to identify domain
3. **Output:** One of 3 labels
   - `SAFETY_REFUSAL` - Ethical violations, requires refusal identification
   - `NON_RAG` - Math, code, reading comprehension (self-contained)
   - `RAG_NECESSITY` - Knowledge-based questions requiring external context

**Implementation:** `src/classification/classify.py`

### 3. Domain-Based Routing

Based on classification, each question is routed with specific configuration:

| Domain | Model | Temperature | Top-P | RAG Enabled |
|--------|-------|------------|-------|-------------|
| SAFETY_REFUSAL | Large | 0.2 | 0.3 | ❌ No |
| NON_RAG | Small | 0.3 | 0.85 | ✅ Yes |
| RAG_NECESSITY | Small | 0.2 | 0.8 | ✅ Yes |

**Configuration:** `src/utils/prompts_config.py`

### 4. RAG Retrieval (Conditional)

**When:** Domain is `NON_RAG` or `RAG_NECESSITY`

**Process:**
1. **Embedding:** Convert question to 1024-dim vector (VNPT embedding model)
2. **Hybrid Search:**
   - FAISS: Semantic similarity search
   - BM25: Keyword-based search
   - Combine with configurable weights (default: 0.5/0.5)
3. **Re-ranking (Optional):** Cross-encoder scores for precision
4. **Selection:** Top-K most relevant chunks (default: K=3)
5. **Output:** Concatenated context string

**Implementation:** `src/rag/retriever.py`

### 5. Prompt Construction

**Components:**
- **System Prompt:** Domain-specific instruction (`src/utils/prompts_config.py`)
- **Context:** Retrieved chunks (if RAG enabled)
- **Question:** Original question text
- **Choices:** Formatted options (A, B, C, D)

**Template:**
```
System: [Domain-specific instructions]

Context: [Retrieved knowledge - if applicable]

Question: [Question text]
A. [Choice 1]
B. [Choice 2]
C. [Choice 3]
D. [Choice 4]
```

**Implementation:** `src/utils/prompt.py`

### 6. LLM Invocation

**Provider:** VNPT AI API (`src/providers/vnpt.py`)

**Features:**
- **Model Selection:** Based on domain routing
- **Infinite Retry:** Automatic handling of quota limits with exponential backoff
- **Error Recovery:** Dimension mismatch detection and retry

**API Flow:**
1. Build request with domain-specific hyperparameters
2. Send to VNPT endpoint with authentication
3. Handle errors (429, 5xx) with retry
4. Return raw prediction text

### 7. Answer Extraction

**Process:**
1. Parse LLM response for answer pattern
2. Extract single letter (A, B, C, D)
3. Handle edge cases:
   - Empty responses → Default to "C"
   - Multiple answers → Take first occurrence
   - API errors → Look for refusal in choices
4. Clean and validate

**Implementation:** `src/utils/prediction.py`

### 8. Timing Measurement

**Measurement Points:**
- **Start:** Before classification
- **End:** After answer extraction
- **Precision:** Rounded to 2 decimal places (seconds)

### 9. Output Generation

**Process:**
1. **Collect Results:** All async tasks complete
2. **Sort:** Order by QID (lexicographic)
3. **Write Files:**
   - `submission.csv`: `qid,answer`
   - `submission_time.csv`: `qid,answer,time`

**Output Format:**

**submission.csv:**
```csv
qid,answer
q001,B
q002,A
q003,D
...
```

**submission_time.csv:**
```csv
qid,answer,time
q001,B,0.82
q002,A,0.65
q003,D,1.23
...
```

---

## 🔧 Resource Initialization

### Overview

Resource initialization occurs during Docker build and at runtime to ensure all required models, indices, and credentials are available.

### 1. Docker Build-Time Initialization

**Script:** `process_data.py`  
**Triggered:** During `docker build` (Dockerfile line: `RUN python3 process_data.py`)

**Process:**

```bash
# 1. Check for documents directory
if [ -d "docs/" ]; then
    # 2. Load documents (PDF, TXT, MD, JSON, CSV, XLSX, DOCX)
    # 3. Extract text from each file
    # 4. Chunk text (smart chunking for markdown/tabular)
    # 5. Generate embeddings via VNPT API
    # 6. Build FAISS index (L2 distance)
    # 7. Build BM25 index (Vietnamese tokenization)
    # 8. Save indices to knowledge_base/
fi
```

**Output:**
- `knowledge_base/faiss_index.bin` - Dense vector index (1024-dim)
- `knowledge_base/text_chunks.json` - Original text chunks
- `knowledge_base/bm25_index.pkl` - Sparse keyword index
- `knowledge_base/index_metadata.json` - File tracking for incremental updates

**Skip Condition:** If indices already exist (idempotent)

### 2. Runtime Initialization (inference.sh)

**Step 1: Vector Database Verification**
```bash
python3 process_data.py
```
- Checks if indices exist
- If missing, builds them (allows runtime initialization if not done at build time)
- If present, skips immediately

**Step 2: Load Components**
```python
# inference.py initialization
provider = load_chat_provider(config)          # VNPT API client
faiss_index, bm25_index, text_chunks, cross_encoder = load_rag_components(config)
```

**Components Loaded:**
1. **Chat Provider:**
   - Credentials from `.secret/api-keys.json`
   - Model routing configuration
   - Retry/backoff settings

2. **RAG Components (if RAG_ENABLED=true):**
   - FAISS index (semantic search)
   - BM25 index (keyword search)
   - Text chunks (source documents)
   - Cross-encoder model (for re-ranking)

3. **Semaphore:**
   - Concurrency control (default: 2 parallel requests)

### 3. Credential Loading

**Priority Order:**
1. **Model-specific env vars** (highest priority)
   - `VNPT_LARGE_ACCESS_TOKEN`, `VNPT_LARGE_TOKEN_ID`, `VNPT_LARGE_TOKEN_KEY`
   - `VNPT_SMALL_ACCESS_TOKEN`, `VNPT_SMALL_TOKEN_ID`, `VNPT_SMALL_TOKEN_KEY`
   - `VNPT_EMBEDDING_ACCESS_TOKEN`, `VNPT_EMBEDDING_TOKEN_ID`, `VNPT_EMBEDDING_TOKEN_KEY`

2. **JSON credentials file** (recommended)
   - Location: `.secret/api-keys.json`
   - Auto-mapped by `llmApiName` field
   - Format:
     ```json
     [
       {
         "authorization": "Bearer TOKEN",
         "tokenKey": "KEY",
         "llmApiName": "LLM embeddings",
         "tokenId": "ID"
       },
       { "llmApiName": "LLM large", ... },
       { "llmApiName": "LLM small", ... }
     ]
     ```

3. **Generic env vars** (fallback)
   - `VNPT_ACCESS_TOKEN`, `VNPT_TOKEN_ID`, `VNPT_TOKEN_KEY`

**Implementation:** `src/providers/vnpt.py` → `_get_credentials()`

### 4. Model Downloads

**Cross-Encoder (Optional, if RERANK_ENABLED=true):**
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Download:** Automatic via Hugging Face transformers
- **Location:** `~/.cache/huggingface/`
- **When:** First load during `load_rag_components()`

**No pre-trained models required for:**
- VNPT API (cloud-hosted)
- FAISS (index built from scratch)
- BM25 (built from documents)

### 5. Configuration Loading

**Source:** `.env` file + environment variables

**Key Settings:**
```bash
# Providers
CHAT_PROVIDER=vnpt
EMBEDDING_PROVIDER=vnpt

# Routing
DOMAIN_ROUTING_ENABLED=true

# RAG
RAG_ENABLED=true
HYBRID_SEARCH_ENABLED=false
RERANK_ENABLED=false

# Performance
CONCURRENT_REQUESTS=2
```

**Loading:** `inference.py` → `build_config_from_env()`

### 6. Error Handling

**Missing Credentials:**
- Logs warning with specific missing credential
- Suggests configuration paths
- Continues if fallback available

**Missing Indices:**
- Attempts to build at runtime
- If `docs/` missing, disables RAG
- Continues with RAG_ENABLED=false

**API Errors:**
- Infinite retry with exponential backoff (if `VNPT_INFINITE_RETRY=true`)
- Max delay: 300 seconds
- Handles 429, 5xx, dimension errors

### 7. Validation

**Pre-flight Checks:**
- [x] Credentials loaded (access token, token ID, token key)
- [x] RAG indices present (if RAG_ENABLED=true)
- [x] Input file exists and valid JSON
- [x] Output directory writable

**Runtime Validation:**
- Embedding dimension consistency (1024-dim)
- Answer format (single letter A-D)
- QID uniqueness

---

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