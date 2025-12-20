# VNPT AI Water Margin - Submission

**Multiple-choice QA system with RAG and intelligent domain-based routing**

## 🚀 Quick Submission

```bash
# 1. Build Docker image
docker build -t username/vnpt-ai-water-margin:latest .

# 2. Test locally
docker run --gpus all \
  -v $(pwd)/data/test.json:/code/private_test.json \
  username/vnpt-ai-water-margin:latest

# 3. Push to DockerHub
docker push username/vnpt-ai-water-margin:latest
```

## 📋 System Overview

The system processes multiple-choice questions using:
- **Domain-based Routing**: Questions are classified and routed to optimal models (small/large) with appropriate configurations
- **RAG (Retrieval-Augmented Generation)**: Knowledge retrieval from document corpus using hybrid search (FAISS + BM25)
- **Parallel Processing**: Efficient async processing with configurable concurrency
- **Production Features**: Infinite retry with exponential backoff, automatic checkpoint/resume

## 🔧 Database Preprocessing

### What Gets Preprocessed

The system builds a **vector database** from documents in the `docs/` directory to enable knowledge retrieval during inference.

**Supported Formats:**
- PDF (`.pdf`) - Research papers, books
- Text (`.txt`) - Plain text
- Markdown (`.md`) - Documentation with structure-aware chunking
- JSON (`.json`) - Structured data
- CSV (`.csv`) - Tabular data with header preservation
- Excel (`.xlsx`) - Spreadsheets
- Word (`.docx`) - Documents

### Preprocessing Steps

**Script:** `process_data.py`

**Process:**

1. **Document Loading**
   - Scans `docs/` directory recursively
   - Loads all supported file types
   - Extracts raw text from each document

2. **Smart Chunking**
   - **Text files**: LangChain's RecursiveCharacterTextSplitter (semantic coherence)
   - **Markdown**: Preserves headers, lists, code blocks
   - **Tabular data** (JSON/CSV/XLSX): Preserves row integrity and headers
   - **Configuration**: Chunk size: 500 chars, overlap: 50 chars

3. **Embedding Generation**
   - Converts each chunk to 1024-dim vector via VNPT embedding API
   - Model: `vnptai_hackathon_embedding`
   - Retry logic for API failures

4. **Index Building**
   - **FAISS Index** (`faiss_index.bin`): Dense semantic search using L2 distance
   - **BM25 Index** (`bm25_index.pkl`): Sparse keyword search with Vietnamese tokenization
   - **Text Chunks** (`text_chunks.json`): Original chunk text for context generation

5. **Persistence**
   - All indices saved to `knowledge_base/` directory
   - Idempotent: Skips rebuild if indices already exist
   - Metadata tracking for incremental updates

### When Preprocessing Occurs

**Docker Build Time (Recommended):**
```dockerfile
# Dockerfile line 46
RUN python3 process_data.py || echo "Vector DB initialization skipped"
```
- Executes during `docker build`
- Pre-builds indices for faster inference
- Falls back gracefully if documents unavailable

**Runtime (Fallback):**
```bash
# inference.sh - Step 1
python3 process_data.py
```
- Checks if indices exist
- Builds if missing (allows runtime initialization)
- Skips if already present (instant startup)

### Configuration

**Environment Variables (.env):**
```bash
# Documents directory
RETRIEVE_DOCS_DIR=docs

# Chunking
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# Embedding
EMBEDDING_PROVIDER=vnpt
EMBEDDING_MODEL_NAME=vnptai_hackathon_embedding
EMBEDDING_DIM=1024
```

### Verification

**Check indices exist:**
```bash
ls knowledge_base/
# Expected output:
# faiss_index.bin       # Dense vector index
# text_chunks.json      # Original chunks
# bm25_index.pkl        # Keyword index
# index_metadata.json   # File tracking
```

## 🎯 Inference Pipeline

### Execution Flow

```mermaid
sequenceDiagram
    participant D as Docker Container
    participant P as process_data.py
    participant I as inference.py
    participant C as Classifier
    participant R as RAG System
    participant L as LLM Provider
    
    D->>P: 1. Initialize vector DB
    P->>P: Check if indices exist
    alt Already exists
        P->>P: Skip build
    else Missing
        P->>R: Build FAISS + BM25
        R->>P: Indices ready
    end
    
    D->>I: 2. Run inference
    I->>I: Load /code/private_test.json
    
    par For each question (parallel)
        I->>C: Classify domain
        C->>I: Label (SAFETY_REFUSAL / NON_RAG / RAG_NECESSITY)
        
        alt Requires RAG
            I->>R: Retrieve context
            R->>I: Top-K chunks
        end
        
        I->>L: Call LLM (domain config)
        L->>I: Raw prediction
        I->>I: Parse answer + timing
    end
    
    I->>I: Sort by QID
    I->>D: Write submission.csv
    I->>D: Write submission_time.csv
```

### Domain-Based Routing

Questions are automatically classified and routed:

| Domain | Model | Temp | RAG | Use Case |
|--------|-------|------|-----|----------|
| **SAFETY_REFUSAL** | Small | 0.3 | No | Ethical/legal violations |
| **NON_RAG** | Large | 0.7 | No | Math, code, reading comprehension |
| **RAG_NECESSITY** | Large | 0.5 | Yes | Knowledge-based questions |

### RAG Retrieval Process

**When RAG is enabled:**

1. **Query Embedding**: Convert question to 1024-dim vector
2. **Hybrid Search** (configurable):
   - FAISS: Semantic similarity search
   - BM25: Keyword-based search
   - Weighted combination (default: 50/50)
3. **Re-ranking** (optional): Cross-encoder scoring for precision
4. **Top-K Selection**: Default K=3 most relevant chunks
5. **Context Formatting**: Concatenated chunks injected into prompt

**Configuration:**
```bash
RAG_ENABLED=true
HYBRID_SEARCH_ENABLED=true
SEMANTIC_WEIGHT=0.5
KEYWORD_WEIGHT=0.5
TOP_K_RAG=3

# Optional re-ranking
RERANK_ENABLED=false
RERANK_TOP_K=10
```

## 📊 Output Format

### submission.csv
```csv
qid,answer
q001,B
q002,A
q003,D
```

### submission_time.csv
```csv
qid,answer,time
q001,B,0.82
q002,A,0.65
q003,D,1.23
```

Both files are:
- Sorted by QID (lexicographic order)
- Single-letter answers (A, B, C, D)
- Timing in seconds (2 decimal places)

## 🔐 Credentials Setup

**Create `.secret/api-keys.json`:**

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

**Automatic Loading:** Credentials are auto-mapped from JSON during initialization.

## 📁 Project Structure

```
VNPT_AI_Water_Margin/
├── .secret/
│   └── api-keys.json           # Credentials (auto-loaded)
├── docs/                       # Knowledge base documents
├── knowledge_base/             # Generated indices
│   ├── faiss_index.bin         # Dense semantic search
│   ├── bm25_index.pkl          # Sparse keyword search
│   └── text_chunks.json        # Processed chunks
├── src/
│   ├── rag/                    # RAG system
│   │   ├── build_index.py      # Index builder
│   │   ├── loader.py           # Load indices
│   │   └── retriever.py        # Hybrid search
│   ├── classification/         # Domain classifier
│   ├── providers/              # LLM providers (VNPT, OpenAI, Ollama)
│   ├── core/                   # Processing core
│   └── utils/                  # Utilities
├── process_data.py             # Vector DB initialization
├── inference.py                # Main inference pipeline
├── inference.sh                # Docker entry point
├── Dockerfile                  # Container definition (CUDA 12.2)
├── requirements.txt            # Python dependencies
└── .env                        # Configuration
```

## ⚙️ Key Configuration

**`.env` file:**

```bash
# Providers
CHAT_PROVIDER=vnpt
EMBEDDING_PROVIDER=vnpt

# Models
MODEL_NAME=vnptai-hackathon-large
SMALL_MODEL_NAME=vnptai-hackathon-small
EMBEDDING_MODEL_NAME=vnptai_hackathon_embedding

# Routing
DOMAIN_ROUTING_ENABLED=true

# RAG
RAG_ENABLED=true
RETRIEVE_DOCS_DIR=docs
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# Performance
CONCURRENT_REQUESTS=2
VNPT_INFINITE_RETRY=true
VNPT_RETRY_INITIAL_DELAY=5
VNPT_RETRY_MAX_DELAY=300
```

## 🚀 Technical Features

### Production-Ready
- ✅ **Infinite Retry**: Automatic quota limit handling with exponential backoff
- ✅ **Checkpoint/Resume**: Automatic progress saving on interruption
- ✅ **Error Recovery**: Dimension mismatch detection and retry
- ✅ **Concurrent Processing**: Async parallel execution with semaphore control

### Intelligent Retrieval
- ✅ **Hybrid Search**: FAISS (semantic) + BM25 (keyword) with configurable weights
- ✅ **Smart Chunking**: Structure-aware processing for markdown and tabular data
- ✅ **Content Filtering**: Automatic removal of irrelevant technical content
- ✅ **Vietnamese Support**: PyVi tokenization for BM25

### Optimizations
- ✅ **Build-time Indexing**: Pre-built vector DB in Docker image
- ✅ **Idempotent Processing**: Skip existing work
- ✅ **Minimal Dependencies**: Only essential packages
- ✅ **CUDA 12.2**: GPU acceleration support

## 📝 Development

### Local Testing

```bash
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure credentials
# Create .secret/api-keys.json (see above)

# 3. Build vector database
python src/rag/build_index.py

# 4. Run inference
python main.py
```

### Docker Testing

```bash
# Build
docker build -t vnpt-test .

# Run with local test file
docker run --gpus all \
  -v $(pwd)/data/test.json:/code/private_test.json \
  vnpt-test

# Check outputs
docker run vnpt-test cat submission.csv
```

## 📖 Additional Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [DOCKER_SUBMISSION.md](DOCKER_SUBMISSION.md) - Submission guide
- [docs/domain_routing_config.md](docs/domain_routing_config.md) - Routing configuration
- [docs/credentials.md](docs/credentials.md) - Credential management
- [docs/infinite_retry.md](docs/infinite_retry.md) - Retry mechanism

---

**Contact:** VNPT AI Water Margin Team  
**Version:** 1.0  
**License:** Competition Submission