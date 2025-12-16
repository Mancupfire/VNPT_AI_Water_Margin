"""Single item processing logic."""

import asyncio
from typing import Dict, Any, Optional
import faiss
from sentence_transformers import CrossEncoder

from src.utils.prompt import format_prompt
from src.utils.prediction import clean_prediction
from src.rag.retriever import retrieve_context
from .config import DEFAULT_CONFIG, merge_config
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


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
        # ======================================================================
        # DOMAIN-BASED ROUTING (Optional - enable via DOMAIN_ROUTING_ENABLED)
        # ======================================================================
        if config.get("DOMAIN_ROUTING_ENABLED", True):
            # Get predicted domain from item (defaults to RAG_NECESSITY if not specified)
            predicted_domain = item.get('predicted_domain', 'RAG_NECESSITY')
            
            # Import routing configuration
            from src.utils.prompts_config import (
                get_model_for_domain,
                get_llm_params_for_domain,
                should_use_rag_for_domain
            )
            
            # Override model name based on domain
            domain_model = get_model_for_domain(predicted_domain)
            original_model = config.get("MODEL_NAME")
            config["MODEL_NAME"] = domain_model
            
            # Override LLM parameters based on domain
            domain_params = get_llm_params_for_domain(predicted_domain)
            original_params = config.get("PAYLOAD_HYPERPARAMS", {})
            config["PAYLOAD_HYPERPARAMS"] = domain_params
            
            # Determine if RAG should be used for this domain
            domain_rag_enabled = should_use_rag_for_domain(predicted_domain) and config.get("RAG_ENABLED", False)
            
            # Log routing decision
            logger.info(
                f"[ROUTING] QID {item.get('qid')}: Domain={predicted_domain} | "
                f"Model={domain_model} | "
                f"Temp={domain_params.get('temperature')} | "
                f"TopP={domain_params.get('top_p')} | "
                f"RAG={'YES' if domain_rag_enabled else 'NO'}"
            )
        else:
            # Domain routing disabled - use default behavior
            predicted_domain = None
            original_model = None
            original_params = config.get("PAYLOAD_HYPERPARAMS", {})
            domain_rag_enabled = config.get("RAG_ENABLED", False)
            
            # Log default behavior
            logger.info(
                f"[DEFAULT] QID {item.get('qid')}: Routing disabled | "
                f"Model={config.get('MODEL_NAME')} | "
                f"Temp={original_params.get('temperature')} | "
                f"RAG={'YES' if domain_rag_enabled else 'NO'}"
            )
        
        context = None
        
        # Retrieve context if RAG is enabled (either domain-specific or global)
        if domain_rag_enabled and text_chunks is not None:
            question_text = item.get("question", "")
            if question_text:
                logger.debug(f"[RAG] QID {item.get('qid')}: Retrieving context...")
                context = await retrieve_context(
                    question_text,
                    provider,
                    config,
                    faiss_index,
                    bm25_index,
                    text_chunks,
                    cross_encoder
                )
                if context:
                    context_preview = context[:100] + "..." if len(context) > 100 else context
                    logger.debug(f"[RAG] QID {item.get('qid')}: Retrieved {len(context)} chars: {context_preview}")
        
        # Format prompt with domain-specific system prompt (if routing enabled)
        messages = format_prompt(item, context, domain=predicted_domain)
        prediction_text = await call_llm_async(provider, messages, config)
        
        # Restore original configuration (if routing was enabled)
        if config.get("DOMAIN_ROUTING_ENABLED", True) and original_model:
            config["MODEL_NAME"] = original_model
            config["PAYLOAD_HYPERPARAMS"] = original_params
        
        # Sleep if configured
        sleep_time = config.get("SLEEP_TIME", DEFAULT_CONFIG['SLEEP_TIME'])
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        # Parse and clean prediction
        clean_pred = _extract_answer(prediction_text, item)
        
        logger.info(f"[ANSWER] QID {item.get('qid')}: Predicted = {clean_pred}")
        
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
