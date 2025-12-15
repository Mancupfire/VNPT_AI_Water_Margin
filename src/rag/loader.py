"""RAG component loading utilities."""

import os
import json
import pickle
import faiss
from sentence_transformers import CrossEncoder
from typing import Dict, Any, Optional, Tuple, List


def load_rag_components(config: Dict[str, Any]) -> Tuple[
    Optional[faiss.Index],
    Optional[Any],
    Optional[List[str]],
    Optional[CrossEncoder]
]:
    """
    Load all RAG components: FAISS index, BM25 index, text chunks, and cross-encoder.
    
    Args:
        config: Configuration dictionary with paths and settings
        
    Returns:
        Tuple of (faiss_index, bm25_index, text_chunks, cross_encoder)
        Returns None for each component if loading fails or RAG is disabled
    """
    if not config.get("RAG_ENABLED"):
        return None, None, None, None
    
    faiss_index = None
    bm25_index = None
    text_chunks = None
    cross_encoder = None
    
    try:
        # Load text chunks (required)
        chunks_path = config.get("TEXT_CHUNKS_PATH")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
        
        # Load FAISS index (for dense/semantic search)
        faiss_path = config.get("FAISS_INDEX_PATH")
        if os.path.exists(faiss_path):
            faiss_index = faiss.read_index(faiss_path)
            print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors.")
        else:
            print(f"⚠️  FAISS index not found at {faiss_path}")
        
        # Load BM25 index (for sparse/keyword search)
        if config.get("HYBRID_SEARCH_ENABLED"):
            bm25_path = config.get("BM25_INDEX_PATH")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
                print("✅ Loaded BM25 index.")
            else:
                print(f"⚠️  BM25 index not found at {bm25_path}")
        
        # Load cross-encoder for re-ranking
        if config.get("RERANK_ENABLED"):
            model_name = config.get("CROSS_ENCODER_MODEL")
            cross_encoder = CrossEncoder(model_name)
            print(f"✅ Loaded Cross-Encoder model: {model_name}")
            
    except Exception as e:
        print(f"❌ Error loading RAG components: {e}. RAG will be disabled.")
        # Disable RAG in config
        config["RAG_ENABLED"] = False
        return None, None, None, None
    
    return faiss_index, bm25_index, text_chunks, cross_encoder
