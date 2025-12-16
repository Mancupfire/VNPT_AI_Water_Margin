"""Configuration management and defaults."""

import os
from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-small"),
    "CONCURRENT_REQUESTS": int(os.getenv("CONCURRENT_REQUESTS", "2")),
    "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "90")),
    "PROVIDER": os.getenv("PROVIDER", "vnpt"),
    "DOMAIN_ROUTING_ENABLED": os.getenv("DOMAIN_ROUTING_ENABLED", "true").lower() == "true",
    "PAYLOAD_HYPERPARAMS": {
        "temperature": 0.5,
        "top_p": 0.7,
        "max_completion_tokens": 2048,
        "n": 1,
        "seed": 416,
    },
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


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()
    if override:
        merged.update(override)
    return merged
