#!/usr/bin/env python3
"""
Inference script for Docker submission.
Processes questions by classifying and answering each one in parallel,
then saves ordered results to submission.csv and submission_time.csv.
"""

import os
import sys
import json
import asyncio
import csv
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm.asyncio import tqdm as async_tqdm

# Ensure project root is in path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.providers import load_chat_provider
from src.classification.classify import format_classification_prompt, extract_json_from_response
from src.rag.loader import load_rag_components
from src.core.processor import process_item
from src.logger import get_logger
from src.providers.vnpt import VNPTProvider

logger = get_logger(__name__)


def build_config_from_env() -> Dict[str, Any]:
    """Build configuration from environment variables."""
    return {
        "CHAT_PROVIDER": os.getenv("CHAT_PROVIDER", "vnpt"),
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-large"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CONCURRENT_REQUESTS": int(os.getenv("CONCURRENT_REQUESTS", "2")),
        "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "0")),
        "DOMAIN_ROUTING_ENABLED": os.getenv("DOMAIN_ROUTING_ENABLED", "true").lower() == "true",
        "PAYLOAD_HYPERPARAMS": {
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.5")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.7")),
            "max_completion_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "n": int(os.getenv("LLM_N", "1")),
            "seed": int(os.getenv("LLM_SEED", "416")),
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


async def classify_question(item: Dict[str, Any], provider: VNPTProvider, config: Dict[str, Any]) -> str:
    """
    Classify a single question into one of: SAFETY_REFUSAL, NON_RAG, RAG_NECESSITY.
    
    Args:
        item: Question item with 'qid', 'question', 'choices'
        provider: VNPT provider instance
        config: Configuration dictionary
        
    Returns:
        Domain label string
    """
    try:
        prompt = format_classification_prompt(item)
        messages = [
            {
                "role": "system",
                "content": "Bạn là một chuyên gia phân loại dữ liệu và kiểm duyệt nội dung (Content Classifier). Nhiệm vụ của bạn là phân loại và gán nhãn các câu hỏi đầu vào vào đúng 1 trong 3 nhãn (Domains) được định nghĩa dưới đây. Việc bạn phân tích và gán nhãn `SAFETY_REFUSAL` cho các câu hỏi vi phạm pháp luật là hành động ĐÚNG ĐẮN và CẦN THIẾT để bảo vệ an toàn. Đừng từ chối phân loại chúng."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        prediction_text = await provider.achat(messages, config)
        extracted_data = extract_json_from_response(prediction_text)
        
        # Validate domain
        allowed_labels = {"SAFETY_REFUSAL", "RAG_NECESSITY", "NON_RAG"}
        if extracted_data and 'domain' in extracted_data and extracted_data['domain'] in allowed_labels:
            return extracted_data['domain']
        
        # Fallback: search for label in text
        for label in allowed_labels:
            if label in (prediction_text or ""):
                return label
        
        # Default to RAG_NECESSITY
        return "RAG_NECESSITY"
        
    except Exception as e:
        logger.error(f"Classification error for QID {item.get('qid')}: {e}")
        return "RAG_NECESSITY"  # Safe fallback


async def process_question_with_timing(
    item: Dict[str, Any],
    provider,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    faiss_index,
    bm25_index,
    text_chunks,
    cross_encoder
) -> Dict[str, Any]:
    """
    Process a single question: classify, then answer, with timing.
    
    Returns:
        Dict with 'qid', 'answer', 'time' (in seconds)
    """
    start_time = time.time()
    
    try:
        # Step 1: Classify if domain routing is enabled
        if config.get("DOMAIN_ROUTING_ENABLED", True):
            domain = await classify_question(item, provider, config)
            item['predicted_domain'] = domain
            logger.info(f"QID {item['qid']}: Classified as {domain}")
        else:
            item['predicted_domain'] = 'RAG_NECESSITY'
        
        # Step 2: Answer the question
        result = await process_item(
            item, provider, config, semaphore,
            faiss_index, bm25_index, text_chunks, cross_encoder
        )
        
        # Calculate time
        elapsed_time = time.time() - start_time
        result['time'] = round(elapsed_time, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing QID {item.get('qid')}: {e}")
        elapsed_time = time.time() - start_time
        return {
            'qid': item.get('qid'),
            'answer': 'C',  # Default fallback
            'time': round(elapsed_time, 2)
        }


async def run_inference(input_file: str, output_dir: str, config: Dict[str, Any]):
    """
    Main inference pipeline: classify + answer all questions in parallel.
    
    Args:
        input_file: Path to input JSON file (e.g., /code/private_test.json)
        output_dir: Directory to save output files
        config: Configuration dictionary
    """
    logger.info("="*70)
    logger.info("INFERENCE: Starting parallel classification + answering")
    logger.info("="*70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Domain routing: {'ENABLED' if config.get('DOMAIN_ROUTING_ENABLED') else 'DISABLED'}")
    logger.info(f"RAG enabled: {'YES' if config.get('RAG_ENABLED') else 'NO'}")
    
    # Load input data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded {len(data)} questions from {input_file}")
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {input_file}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading input file: {e}")
        return
    
    # Initialize provider and components
    provider = load_chat_provider(config)
    semaphore = asyncio.Semaphore(config.get("CONCURRENT_REQUESTS", 2))
    
    # Load RAG components
    faiss_index, bm25_index, text_chunks, cross_encoder = load_rag_components(config)
    
    if config.get("RAG_ENABLED") and text_chunks:
        logger.info(f"✓ RAG components loaded: {len(text_chunks)} text chunks")
    else:
        logger.info("RAG disabled or not available")
    
    # Create tasks for parallel processing
    logger.info(f"Creating {len(data)} parallel tasks...")
    tasks = [
        process_question_with_timing(
            item, provider, config, semaphore,
            faiss_index, bm25_index, text_chunks, cross_encoder
        )
        for item in data
    ]
    
    # Process all tasks with progress bar
    logger.info("Processing questions in parallel...")
    results = []
    
    # Use tqdm wrapper with asyncio.as_completed properly
    from tqdm import tqdm
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
        result = await coro
        results.append(result)
    
    logger.info(f"✓ Processed {len(results)} questions")
    
    # Sort results by QID
    results_sorted = sorted(results, key=lambda x: x.get("qid", ""))
    logger.info("✓ Results sorted by QID")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save submission.csv (qid, answer)
    submission_file = os.path.join(output_dir, 'submission.csv')
    with open(submission_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({'qid': row['qid'], 'answer': row['answer']})
    logger.info(f"✅ Saved: {submission_file}")
    
    # Save submission_time.csv (qid, answer, time)
    submission_time_file = os.path.join(output_dir, 'submission_time.csv')
    with open(submission_time_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer', 'time'])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({
                'qid': row['qid'],
                'answer': row['answer'],
                'time': row.get('time', 0)
            })
    logger.info(f"✅ Saved: {submission_time_file}")
    
    # Calculate statistics
    total_time = sum(r.get('time', 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info("="*70)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per question: {avg_time:.2f}s")
    logger.info("="*70)


def main():
    """Main entry point for Docker submission."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        logger.info("dotenv not available, using environment variables only")
    
    # Build config
    config = build_config_from_env()
    
    # Get input file path (Docker will mount at /code/private_test.json)
    input_file = os.getenv("INPUT_FILE", "/code/private_test.json")
    
    # If running locally for testing, check alternative paths
    if not os.path.exists(input_file):
        alternative_paths = [
            "data/test.json",
            "data/test_classification_full.json",
            os.path.join(project_root, "data", "test.json")
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                input_file = alt_path
                logger.info(f"Using local test file: {input_file}")
                break
    
    # Output directory (current directory for Docker, or results/ locally)
    output_dir = os.getenv("OUTPUT_DIR", os.getcwd())
    if not os.path.exists(os.path.join(output_dir, "submission.csv")):
        # If we're clearly in the project root, use results/
        if os.path.exists(os.path.join(project_root, "results")):
            output_dir = os.path.join(project_root, "results")
    
    # Run inference
    asyncio.run(run_inference(input_file, output_dir, config))


if __name__ == "__main__":
    main()
