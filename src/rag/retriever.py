"""Context retrieval using hybrid search and re-ranking."""

import asyncio
import numpy as np
import faiss
from pyvi import ViTokenizer
from typing import Dict, Any, Optional, List
from sentence_transformers import CrossEncoder


async def retrieve_context(
    question: str,
    provider,
    config: Dict[str, Any],
    faiss_index: Optional[faiss.Index],
    bm25_index: Optional[Any],
    text_chunks: Optional[List[str]],
    cross_encoder: Optional[CrossEncoder]
) -> Optional[str]:
    """
    Retrieve relevant context for a question using RAG.
    
    Supports multiple retrieval strategies:
    - Hybrid search (FAISS + BM25)
    - Dense search (FAISS only)
    - Re-ranking with cross-encoder
    
    Args:
        question: The question text to retrieve context for
        provider: Provider for embedding generation
        config: Configuration dictionary
        faiss_index: FAISS index for semantic search
        bm25_index: BM25 index for keyword search
        text_chunks: List of text chunks from knowledge base
        cross_encoder: Cross-encoder model for re-ranking
        
    Returns:
        Retrieved context as a string, or None if retrieval fails
    """
    if not question or text_chunks is None:
        return None
    
    # Determine how many chunks to retrieve initially
    retrieval_top_k = (
        config.get("RERANK_TOP_K", 10) 
        if config.get("RERANK_ENABLED") 
        else config.get("TOP_K_RAG", 3)
    )
    
    retrieved_chunks = []
    
    # Choose retrieval strategy
    if config.get("HYBRID_SEARCH_ENABLED") and bm25_index is not None and faiss_index is not None:
        retrieved_chunks = await _hybrid_search(
            question, provider, config, faiss_index, bm25_index, text_chunks, retrieval_top_k
        )
    elif faiss_index is not None:
        retrieved_chunks = await _dense_search(
            question, provider, config, faiss_index, text_chunks, retrieval_top_k
        )
    
    # Apply re-ranking if enabled
    if config.get("RERANK_ENABLED") and cross_encoder is not None and retrieved_chunks:
        final_chunks = await _rerank_chunks(
            question, retrieved_chunks, cross_encoder, config.get("TOP_K_RAG", 3)
        )
        return "\n".join(final_chunks)
    
    return "\n".join(retrieved_chunks)


async def _hybrid_search(
    question: str,
    provider,
    config: Dict[str, Any],
    faiss_index: faiss.Index,
    bm25_index: Any,
    text_chunks: List[str],
    top_k: int
) -> List[str]:
    """
    Perform hybrid search combining FAISS (semantic) and BM25 (keyword) scores.
    
    Args:
        question: Query text
        provider: Provider for embeddings
        config: Configuration with weights
        faiss_index: FAISS index
        bm25_index: BM25 index
        text_chunks: Text corpus
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved text chunks
    """
    # Tokenize for BM25
    tokenized_query = ViTokenizer.tokenize(question).split()
    
    loop = asyncio.get_running_loop()
    
    # Get BM25 scores (CPU-bound, run in executor)
    bm25_scores = await loop.run_in_executor(None, bm25_index.get_scores, tokenized_query)
    
    # Get FAISS scores
    question_embedding_list = await provider.aembed(
        [question], 
        {"MODEL_NAME": "vnptai_hackathon_embedding"}
    )
    question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
    
    # FAISS search (CPU-bound, run in executor)
    D, I = await loop.run_in_executor(
        None, 
        lambda: faiss_index.search(question_embedding, len(text_chunks))
    )
    
    # Normalize BM25 scores
    bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
    if bm25_max - bm25_min == 0:
        bm25_scores_norm = np.zeros_like(bm25_scores)
    else:
        bm25_scores_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    
    # Normalize FAISS scores (distances, so invert)
    faiss_min, faiss_max = np.min(D[0]), np.max(D[0])
    if faiss_max - faiss_min == 0:
        faiss_scores_norm = np.zeros_like(D[0])
    else:
        faiss_scores_norm = 1 - (D[0] - faiss_min) / (faiss_max - faiss_min)
    
    # Combine scores with weights
    semantic_weight = config.get("SEMANTIC_WEIGHT", 0.5)
    keyword_weight = config.get("KEYWORD_WEIGHT", 0.5)
    hybrid_scores = (semantic_weight * faiss_scores_norm) + (keyword_weight * bm25_scores_norm)
    
    # Get top-k chunks by hybrid score
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    retrieved_chunks = [text_chunks[i] for i in sorted_indices[:top_k]]
    
    return retrieved_chunks


async def _dense_search(
    question: str,
    provider,
    config: Dict[str, Any],
    faiss_index: faiss.Index,
    text_chunks: List[str],
    top_k: int
) -> List[str]:
    """
    Perform dense search using FAISS only (semantic similarity).
    
    Args:
        question: Query text
        provider: Provider for embeddings
        config: Configuration
        faiss_index: FAISS index
        text_chunks: Text corpus
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved text chunks
    """
    # Generate question embedding
    question_embedding_list = await provider.aembed(
        [question],
        {"MODEL_NAME": "vnptai_hackathon_embedding"}
    )
    question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
    
    # FAISS search (CPU-bound, run in executor)
    loop = asyncio.get_running_loop()
    D, I = await loop.run_in_executor(
        None,
        lambda: faiss_index.search(question_embedding, top_k)
    )
    
    # Get chunks by indices
    retrieved_chunks = [
        text_chunks[idx] 
        for idx in I[0] 
        if idx >= 0 and idx < len(text_chunks)
    ]
    
    return retrieved_chunks


async def _rerank_chunks(
    question: str,
    chunks: List[str],
    cross_encoder: CrossEncoder,
    final_k: int
) -> List[str]:
    """
    Re-rank retrieved chunks using a cross-encoder model.
    
    Args:
        question: Query text
        chunks: Retrieved chunks to re-rank
        cross_encoder: Cross-encoder model
        final_k: Number of final chunks to return
        
    Returns:
        Top-k re-ranked chunks
    """
    # Create pairs for cross-encoder
    pairs = [[question, chunk] for chunk in chunks]
    
    # Score pairs (CPU-bound, but fast enough)
    scores = cross_encoder.predict(pairs)
    
    # Sort by score descending
    sorted_chunks = [
        chunk 
        for _, chunk in sorted(zip(scores, chunks), reverse=True)
    ]
    
    return sorted_chunks[:final_k]
