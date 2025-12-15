"""Single item processing logic."""

import asyncio
from typing import Dict, Any, Optional
import faiss
from sentence_transformers import CrossEncoder

from src.utils.prompt import format_prompt
from src.utils.prediction import clean_prediction
from src.RAG.retriever import retrieve_context
from .config import DEFAULT_CONFIG, merge_config


async def call_llm_async(
    provider, 
    messages: list[Dict[str, Any]], 
    config: Dict[str, Any] | None = None
) -> str:
    """
    Dispatch async chat request to provider and return response text.
    
    Args:
        provider: Chat provider instance
        messages: List of message dictionaries
        config: Configuration dictionary
        
    Returns:
        Response text from the LLM
    """
    cfg = merge_config(DEFAULT_CONFIG, config)
    return await provider.achat(messages, cfg)


async def process_item(
    item: Dict[str, Any],
    provider,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    faiss_index: Optional[faiss.Index] = None,
    bm25_index: Optional[Any] = None,
    text_chunks: Optional[list] = None,
    cross_encoder: Optional[CrossEncoder] = None
) -> Dict[str, Any]:
    """
    Process a single question item: retrieve context, call LLM, parse answer.
    
    Args:
        item: Question dictionary with 'qid', 'question', 'choices', optional 'answer'
        provider: Chat/Embedding provider
        config: Configuration dictionary
        semaphore: Asyncio semaphore for concurrency control
        faiss_index: Optional FAISS index for RAG
        bm25_index: Optional BM25 index for hybrid search
        text_chunks: Optional text chunks for RAG
        cross_encoder: Optional cross-encoder for re-ranking
        
    Returns:
        Result dictionary with 'qid', 'answer', 'prediction_raw', and optional 'ground_truth', 'is_correct'
    """
    async with semaphore:
        context = None
        
        # Retrieve context if RAG is enabled
        if config.get("RAG_ENABLED") and text_chunks is not None:
            question_text = item.get("question", "")
            if question_text:
                context = await retrieve_context(
                    question_text,
                    provider,
                    config,
                    faiss_index,
                    bm25_index,
                    text_chunks,
                    cross_encoder
                )
        
        # Format prompt and call LLM
        messages = format_prompt(item, context)
        prediction_text = await call_llm_async(provider, messages, config)
        
        # Sleep if configured
        sleep_time = config.get("SLEEP_TIME", DEFAULT_CONFIG['SLEEP_TIME'])
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        # Parse and clean prediction
        clean_pred = _extract_answer(prediction_text, item)
        
        # Build result
        result = {
            "qid": item.get('qid'),
            "answer": clean_pred,
            "prediction_raw": prediction_text.replace('\n', ' ') if prediction_text else ""
        }
        
        # Add ground truth comparison if available
        if 'answer' in item:
            ground_truth = item.get('answer', '')
            is_correct = (clean_pred.upper() == ground_truth.upper()) if ground_truth else False
            result["ground_truth"] = ground_truth
            result["is_correct"] = is_correct
        
        return result


def _extract_answer(prediction_text: str, item: Dict[str, Any]) -> str:
    """
    Extract answer from prediction text with special handling for edge cases.
    
    Args:
        prediction_text: Raw LLM output
        item: Question item for context
        
    Returns:
        Single letter answer (A, B, C, D, etc.)
    """
    # Handle empty response
    if not prediction_text:
        return "C"
    
    # Handle single letter response
    if len(prediction_text) == 1 and prediction_text.isalpha() and prediction_text.isupper():
        return prediction_text
    
    # Handle ethical refusal - look for refusal in choices
    if prediction_text.startswith("Error from VNPT API:"):
        for i, choice in enumerate(item.get('choices', [])):
            if any(phrase in choice for phrase in ["không thể", "xin lỗi", "không cung cấp", "không trả lời"]):
                return chr(ord('A') + i)
    
    # Default extraction
    return clean_prediction(prediction_text)
