# VNPT_AI_Water_Margin

Lightweight runner that queries a VNPT-hosted LLM to answer multiple-choice questions from JSON datasets and writes predictions to CSV/JSON. This project now supports Retrieval-Augmented Generation (RAG) and asynchronous processing for improved performance and accuracy.

Quickstart

1. Create a Python 3.11+ virtual environment and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example` and populate the VNPT credentials:

```pwsh
copy .env.example .env
# then edit .env and fill in the necessary variables
```

3. Run the main runner (synchronous):

```pwsh
python main.py
```

4. Run the asynchronous runner (recommended for performance, see RAG section below):

```pwsh
python async_main.py
```

Retrieval-Augmented Generation (RAG)

To improve the accuracy of LLM responses by providing relevant context from your documents, you can enable RAG:

1.  **Prepare your Knowledge Base:** Place your PDF documents in the `docs/` directory.

2.  **Build the RAG Indices:** Run the following command to process your PDF documents, generate embeddings, and create searchable indices:

    ```pwsh
    python build_index.py
    ```

    This will create the following files in the `knowledge_base` directory:
    - `faiss_index.bin`: The FAISS index for dense, semantic search.
    - `bm25_index.pkl`: The BM25 index for sparse, keyword-based search.
    - `text_chunks.json`: The raw text chunks corresponding to the indexed vectors.

3.  **Enable RAG in the Runner:** Add the following environment variables to your `.env` file to activate RAG and configure its behavior:

    ```dotenv
    RAG_ENABLED=true
    TOP_K_RAG=3  # Optional: Number of top relevant chunks to retrieve (default is 3)
    ```

#### Advanced RAG: Hybrid Search with Re-ranking

For even higher accuracy, you can enable a two-stage retrieval process that uses a hybrid search (combining dense and sparse retrieval) followed by a re-ranker.

1.  **Enable Hybrid Search:** To combine semantic search (FAISS) with keyword search (BM25), add the following to your `.env` file:

    ```dotenv
    HYBRID_SEARCH_ENABLED=true
    SEMANTIC_WEIGHT=0.5 # Optional: Weight for semantic search score (0.0 to 1.0)
    KEYWORD_WEIGHT=0.5  # Optional: Weight for keyword search score (0.0 to 1.0)
    ```

2.  **Enable Re-ranking:** For the highest accuracy, use a `CrossEncoder` model to re-rank the results from the hybrid search.

    ```dotenv
    RERANK_ENABLED=true
    CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 # Optional: Specify a CrossEncoder model
    RERANK_TOP_K=10 # Optional: The number of initial documents to retrieve for re-ranking
    ```

    With `RERANK_ENABLED=true`, the `TOP_K_RAG` variable will determine how many of the re-ranked documents are passed to the LLM.

#### Pre-retrieving Context

If you plan to run the pipeline multiple times on the same dataset, you can pre-retrieve the context for each question to save time.

1.  **Run the Pre-retrieval Script:**
    ```pwsh
    python pre_retrieve.py
    ```
    This will read your input dataset (e.g., `data/test.json`), perform the configured retrieval process (hybrid search, re-ranking, etc.), and save a new dataset (e.g., `data/test_with_context.json`) with a `retrieved_context` field added to each question.

2.  **Use the Pre-retrieved Dataset:** In your `.env` file, set the following:
    ```dotenv
    USE_PRE_RETRIEVED_CONTEXT=true
    ```
    Then, when you run `async_main.py`, make sure to point it to the new dataset with the pre-filled context. The runner will use this context and skip the time-consuming real-time retrieval step.


Logging

The application now includes a structured logging system.
- Logs are printed to the console and also saved to a file in the `logs/` directory (`logs/app.log`).
- You can control the logging verbosity by setting the `LOG_LEVEL` environment variable (e.g., `INFO`, `DEBUG`).
  ```dotenv
  LOG_LEVEL=INFO
  ```

Use different providers

This project supports different providers for both chat completion and embedding generation. You can configure them independently in your `.env` file.

- **`CHAT_PROVIDER`**: `vnpt`, `ollama`, `openai` (default: `vnpt`)
- **`EMBEDDING_PROVIDER`**: `vnpt`, `huggingface` (default: `vnpt`)

**Examples:**

- **VNPT (default):**
  ```dotenv
  CHAT_PROVIDER=vnpt
  EMBEDDING_PROVIDER=vnpt
  VNPT_ACCESS_TOKEN=...
  VNPT_TOKEN_ID=...
  VNPT_TOKEN_KEY=...
  ```

- **Local testing with Ollama and Hugging Face:**
  ```dotenv
  CHAT_PROVIDER=ollama
  MODEL_NAME=gemma2:270m
  OLLAMA_BASE=http://localhost:11434

  EMBEDDING_PROVIDER=huggingface
  HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
  ```

- **OpenAI:**
  ```dotenv
  CHAT_PROVIDER=openai
  MODEL_NAME=gpt-3.5-turbo
  OPENAI_API_KEY=...
  ```

Notes

- Secrets should be provided via environment variables; do not commit credentials to version control.
- For asynchronous processing (`async_main.py`), `SLEEP_TIME` (default 90 seconds, for 40 req/h quota) and `CONCURRENT_REQUESTS` (default 2) are controlled via environment variables.

Docker

Build the container:

```pwsh
docker build -t vnpt-ai-water-margin:latest .
```

Run container (example using env file):

```pwsh
docker run --rm --env-file .env vnpt-ai-water-margin:latest
```

Adapting providers

- Provider implementations are in `src/providers/`. To add a provider, implement a `create(config)` factory that returns a provider instance. Chat providers should have an `achat` method, and embedding providers should have an `aembed` method.