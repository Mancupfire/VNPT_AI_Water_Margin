#!/usr/bin/env python3
"""
Data preprocessing script for Docker submission.
Initializes the vector database (FAISS + BM25 indices) before inference.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.rag.build_index import build_index
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    """Build RAG indices from documents directory."""
    logger.info("="*70)
    logger.info("PROCESS_DATA: Initializing Vector Database")
    logger.info("="*70)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        logger.info("dotenv not available, using environment variables only")
    
    # Configuration
    config = {
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "vnptai_hackathon_embedding"),
        "CHUNK_SIZE": int(os.getenv("RAG_CHUNK_SIZE", "500")),
        "CHUNK_OVERLAP": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        "EMBEDDING_DIM": int(os.getenv("EMBEDDING_DIM", "1024")),
    }
    
    # Paths
    docs_directory = os.getenv("RETRIEVE_DOCS_DIR", "docs")
    if not os.path.isabs(docs_directory):
        docs_directory = os.path.join(project_root, docs_directory)
    
    knowledge_base_dir = os.path.join(project_root, "knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)
    
    faiss_index_file = os.path.join(knowledge_base_dir, "faiss_index.bin")
    text_chunks_file = os.path.join(knowledge_base_dir, "text_chunks.json")
    bm25_index_file = os.path.join(knowledge_base_dir, "bm25_index.pkl")
    
    # Check if documents directory exists
    if not os.path.exists(docs_directory):
        logger.warning(f"Documents directory not found: {docs_directory}")
        logger.warning("RAG will be disabled during inference")
        return
    
    # Check if indices already exist
    if all(os.path.exists(f) for f in [faiss_index_file, text_chunks_file, bm25_index_file]):
        logger.info("✓ Vector database already exists, skipping build")
        logger.info(f"  - FAISS index: {faiss_index_file}")
        logger.info(f"  - Text chunks: {text_chunks_file}")
        logger.info(f"  - BM25 index: {bm25_index_file}")
        return
    
    logger.info(f"Building vector database from: {docs_directory}")
    
    # Build indices
    asyncio.run(
        build_index(
            docs_directory,
            faiss_index_file,
            text_chunks_file,
            bm25_index_file,
            config
        )
    )
    
    logger.info("="*70)
    logger.info("✅ PROCESS_DATA: Vector database initialization complete")
    logger.info("="*70)

if __name__ == "__main__":
    main()
