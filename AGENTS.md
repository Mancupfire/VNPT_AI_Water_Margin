# Agents Overview: VNPT AI Water Margin Project

This project provides a robust and extensible framework for answering multiple-choice questions by leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Its modular architecture allows for flexible integration with various LLM and embedding providers, alongside advanced retrieval strategies.

## Overall Project Architecture

The core of the project revolves around an asynchronous runner (`async_main.py` / `src/async_running.py`) that processes datasets of multiple-choice questions. Key architectural components include:

1.  **Modular Provider System:** Decoupled interfaces for Chat Completion and Embedding generation, allowing different backend services to be used independently.
2.  **Asynchronous Processing:** Utilizes `asyncio` for concurrent execution of LLM requests, significantly improving throughput while respecting API rate limits.
3.  **Advanced RAG Pipeline:** Integrates sophisticated document retrieval and augmentation techniques to provide LLMs with relevant context, thereby enhancing answer accuracy.
4.  **Logging:** A centralized logging system (`src/logger.py`) ensures that all critical operations, errors, and debugging information are recorded for future analysis.

## RAG Pipeline Details

The RAG pipeline is designed to efficiently retrieve and synthesize relevant information from a knowledge base to augment LLM prompts.

1.  **Knowledge Base Construction (`build_index.py`):
    *   **Document Ingestion:** PDF documents from the `docs/` directory are parsed and their text content is extracted.
    *   **Semantic Chunking:** Text is broken down into coherent, semantically meaningful chunks using a recursive splitting strategy (paragraphs, sentences, words).
    *   **Dual Indexing:**
        *   **Dense Index (FAISS):** Embeddings for each text chunk are generated using a configurable embedding provider (e.g., VNPT AI, Hugging Face) and stored in a FAISS vector index for semantic similarity search.
        *   **Sparse Index (BM25):** A BM25 keyword-based index is also created from the tokenized text chunks to capture lexical relevance.

2.  **Context Retrieval (`src/async_running.py`):
    *   **Hybrid Search:** For each question, both dense (FAISS) and sparse (BM25) searches are performed. Their respective scores are normalized and combined using configurable weights (`SEMANTIC_WEIGHT`, `KEYWORD_WEIGHT`) to produce a balanced relevance ranking.
    *   **Re-ranking:** The top documents from the hybrid search are then passed through a `CrossEncoder` model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) for a more fine-grained re-evaluation of their relevance to the question. This two-stage retrieval significantly improves the quality of the context.
    *   **Prompt Augmentation:** The most relevant chunks are then prepended to the user's question, forming an augmented prompt that is sent to the LLM.

3.  **Pre-retrieval Optimization (`pre_retrieve.py`):
    *   To further accelerate repeated evaluations, a `pre_retrieve.py` script allows for offline execution of the full retrieval pipeline. This script generates a new dataset where each question object already contains its `retrieved_context`.
    *   When the main runner is configured with `USE_PRE_RETRIEVED_CONTEXT=true`, it skips the real-time retrieval steps and directly uses the pre-computed context, saving significant processing time.

## Chat Providers

Chat providers are responsible for generating the answer to a multiple-choice question, optionally using context provided by the RAG pipeline.

### 1. VNPT AI

*   **Description:** The primary LLM provider for this project, hosted by VNPT.
*   **Configuration:** Requires `VNPT_ACCESS_TOKEN`, `VNPT_TOKEN_ID`, and `VNPT_TOKEN_KEY`. The specific model (e.g., `vnptai-hackathon-small`, `vnptai-hackathon-large`) can be specified via the `MODEL_NAME` environment variable.
*   **Endpoint:** `/data-service/v1/chat/completions/{model_name}`
*   **Rate Limits:** As documented by VNPT, typically 40-60 requests per hour.

### 2. Ollama

*   **Description:** Supports interaction with local or remote Ollama instances, which host open-source LLMs.
*   **Configuration:** Set `CHAT_PROVIDER=ollama` and `OLLAMA_BASE` (e.g., `http://localhost:11434`) in your `.env` file. The model name for Ollama can be set via `MODEL_NAME`.
*   **Implementation:** Uses an `aiohttp`-based HTTP client to communicate with Ollama's API.

### 3. OpenAI

*   **Description:** Integrates with OpenAI's powerful language models, such as GPT-3.5 and GPT-4.
*   **Configuration:** Set `CHAT_PROVIDER=openai` and `OPENAI_API_KEY` in your `.env` file. The specific OpenAI model can be chosen using the `MODEL_NAME` environment variable (e.g., `gpt-3.5-turbo`).
*   **Implementation:** Uses the `openai` Python library's asynchronous client (`openai.AsyncOpenAI`).

## Embedding Providers

Embedding providers are used by the RAG pipeline to generate vector embeddings for text chunks and questions.

### 1. VNPT AI

*   **Description:** Uses the VNPT AI embedding model.
*   **Configuration:** Requires `VNPT_ACCESS_TOKEN`, `VNPT_TOKEN_ID`, and `VNPT_TOKEN_KEY`. The model name is fixed to `vnptai_hackathon_embedding`.
*   **Endpoint:** `/data-service/vnptai-hackathon-embedding`
*   **Rate Limits:** 500 requests per minute.

### 2. Hugging Face

*   **Description:** Uses a local `sentence-transformers` model from Hugging Face to generate embeddings. This is useful for offline testing and development.
*   **Configuration:** Set `EMBEDDING_PROVIDER=huggingface` in your `.env` file. You can also specify the model to use with `HUGGINGFACE_EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`).

## Key Configuration Variables

These environment variables (typically set in `.env`) control the project's behavior:

*   **`CHAT_PROVIDER`**: Specifies the LLM provider for chat completions.
*   **`EMBEDDING_PROVIDER`**: Specifies the provider for generating embeddings.
*   **`MODEL_NAME`**: The specific model to use with the selected provider (for chat).
*   **`HUGGINGFACE_EMBEDDING_MODEL`**: The specific model to use with the Hugging Face embedding provider.
*   **`RAG_ENABLED`**: Enables or disables the entire RAG pipeline.
*   **`HYBRID_SEARCH_ENABLED`**: Enables or disables the hybrid retrieval (BM25 + FAISS) within RAG.
*   **`SEMANTIC_WEIGHT` / `KEYWORD_WEIGHT`**: Controls the balance between semantic and keyword search in hybrid retrieval.
*   **`RERANK_ENABLED`**: Enables or disables the re-ranking step within RAG.
*   **`CROSS_ENCODER_MODEL`**: The model used for re-ranking.
*   **`USE_PRE_RETRIEVED_CONTEXT`**: When true, the runner uses pre-computed context from the input dataset, skipping real-time retrieval.
*   **`LOG_LEVEL`**: Sets the verbosity of the logging output (e.g., `INFO`, `DEBUG`).
*   **`CONCURRENT_REQUESTS`**: Limits the number of simultaneous LLM requests in asynchronous mode.
*   **`SLEEP_TIME`**: Sets the delay between LLM requests to respect API rate limits.

This detailed overview should help any agent understand the project's capabilities and how to configure them.

