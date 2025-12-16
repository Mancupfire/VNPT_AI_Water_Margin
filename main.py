import os
import asyncio
from src.async_running import process_dataset_async
from src.classification.classify import process_classification_dataset

try:
    # optional: auto-load .env for local development
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _build_config_from_env():
    """
    Build configuration from environment variables.
    
    Note: VNPT credentials (ACCESS_TOKEN, TOKEN_ID, TOKEN_KEY) are automatically
    loaded from .secret/api-keys.json by default. You don't need to set them here
    unless you want to override the JSON file.
    """
    return {
        # Provider Configuration
        "CHAT_PROVIDER": os.getenv("CHAT_PROVIDER", "vnpt"),
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-large"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        
        # Performance and Rate Limiting
        "CONCURRENT_REQUESTS": int(os.getenv("CONCURRENT_REQUESTS", "2")),
        "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "0")),  # Can be 0 with infinite retry
        
        # LLM Hyperparameters (all configurable via .env)
        "PAYLOAD_HYPERPARAMS": {
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.5")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.7")),
            "max_completion_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "n": int(os.getenv("LLM_N", "1")),
            "seed": int(os.getenv("LLM_SEED", "416")),
        },
        
        # RAG Configuration
        "RAG_ENABLED": os.getenv("RAG_ENABLED", "false").lower() == "true",
        "TOP_K_RAG": int(os.getenv("TOP_K_RAG", "3")),
        "FAISS_INDEX_PATH": os.getenv("FAISS_INDEX_PATH", "knowledge_base/faiss_index.bin"),
        "TEXT_CHUNKS_PATH": os.getenv("TEXT_CHUNKS_PATH", "knowledge_base/text_chunks.json"),
        "RERANK_ENABLED": os.getenv("RERANK_ENABLED", "false").lower() == "true",
        "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "RERANK_TOP_K": int(os.getenv("RERANK_TOP_K", "10")),
        "HYBRID_SEARCH_ENABLED": os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true",
        "BM25_INDEX_PATH": os.getenv("BM25_INDEX_PATH", "knowledge_base/bm25_index.pkl"),
        "SEMANTIC_WEIGHT": float(os.getenv("SEMANTIC_WEIGHT", "0.5")),
        "KEYWORD_WEIGHT": float(os.getenv("KEYWORD_WEIGHT", "0.5")),
    }


if __name__ == "__main__":
    config = _build_config_from_env()
    
    # Chạy phân loại test
    # process_classification_dataset(
    #     input_file='data/test.json',
    #     output_file='results/test_classification.json',
    #     config=config
    # )
    
    # Chạy phân loại async (giữ nguyên)
    asyncio.run(
        process_dataset_async(
            input_file='data/test.json',
            output_file=f'results/test_{config.get("CHAT_PROVIDER", "")}_async.csv',
            config=config,
            mode='test'
        )
    )
