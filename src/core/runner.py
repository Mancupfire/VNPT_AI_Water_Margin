"""Main async orchestration for test and validation modes."""

import os
import json
import asyncio
import csv
from tqdm.asyncio import tqdm
from typing import Dict, Any, List

from src.providers import load_chat_provider
from src.utils.progress import load_progress, filter_items, display_progress_info
from src.rag.loader import load_rag_components
from .processor import process_item
from .config import DEFAULT_CONFIG


async def run_test_mode(input_file: str, output_file: str, config: Dict[str, Any]) -> None:
    """
    Run in test mode: processes questions without ground truth, supports resume.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output CSV file
        config: Configuration dictionary
    """
    print(f"--- Processing (test mode): {input_file} -> {output_file} ---")
    
    # Load dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return
    
    # Initialize provider and semaphore
    provider = load_chat_provider(config)
    semaphore = asyncio.Semaphore(config.get("CONCURRENT_REQUESTS", 2))
    
    # Load RAG components
    faiss_index, bm25_index, text_chunks, cross_encoder = load_rag_components(config)
    
    # Load progress (checkpoint/resume)
    processed_qids, existing_results = load_progress(output_file)
    
    # Filter out already processed items
    items_to_process = filter_items(data, processed_qids)
    
    # Display progress info
    display_progress_info(len(processed_qids), len(data), len(items_to_process))
    
    if not items_to_process:
        return
    
    # Create tasks
    tasks = [
        process_item(item, provider, config, semaphore, faiss_index, bm25_index, text_chunks, cross_encoder)
        for item in items_to_process
    ]
    
    # Process with incremental saving
    results = existing_results.copy()
    fieldnames = ["qid", "answer", "prediction_raw"]
    
    with open(output_file, 'a', encoding='utf-8', newline='') as f_result:
        writer_result = csv.DictWriter(f_result, fieldnames=fieldnames)
        
        # Write header if new file
        if not existing_results:
            writer_result.writeheader()
            f_result.flush()
        
        # Process tasks with progress bar
        with tqdm(total=len(items_to_process), desc="Processing") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Write immediately
                writer_result.writerow(result)
                f_result.flush()
                
                pbar.update(1)
    
    print(f"✅ Done. Results saved to: {output_file}")
    
    # Generate submission file
    _save_submission_file(results, output_file)


async def run_validation_mode(input_file: str, output_file: str, config: Dict[str, Any]) -> None:
    """
    Run in validation mode: processes questions with ground truth, calculates accuracy.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output CSV file
        config: Configuration dictionary
    """
    print(f"--- Processing (validation mode): {input_file} -> {output_file} ---")
    
    # Load dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return
    
    # Initialize provider and semaphore
    provider = load_chat_provider(config)
    semaphore = asyncio.Semaphore(config.get("CONCURRENT_REQUESTS", 2))
    
    # Load RAG components
    faiss_index, bm25_index, text_chunks, cross_encoder = load_rag_components(config)
    
    # Create tasks
    tasks = [
        process_item(item, provider, config, semaphore, faiss_index, bm25_index, text_chunks, cross_encoder)
        for item in data
    ]
    
    # Process with incremental saving and accuracy tracking
    results = []
    correct = 0
    total = len(tasks)
    fieldnames = ["qid", "prediction", "ground_truth", "is_correct", "prediction_raw"]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        
        with tqdm(total=total, desc="Processing") as pbar:
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                
                if res.get('is_correct'):
                    correct += 1
                
                # Write immediately
                writer.writerow({
                    "qid": res.get("qid"),
                    "prediction": res.get("answer"),
                    "ground_truth": res.get("ground_truth"),
                    "is_correct": res.get("is_correct"),
                    "prediction_raw": res.get("prediction_raw")
                })
                f.flush()
                
                # Update progress bar with accuracy
                current_acc = correct / len(results) if results else 0
                pbar.set_postfix({"acc": f"{current_acc:.2%}", "correct": correct})
                pbar.update(1)
    
    acc = correct / total if total > 0 else 0.0
    print(f"✅ Done. Results saved to: {output_file}")
    print(f"📊 Accuracy: {acc:.4f} ({correct}/{total})")
    
    # Generate submission file
    _save_submission_file(results, output_file)


async def process_dataset(
    input_file: str,
    output_file: str,
    config: Dict[str, Any] | None = None,
    mode: str | None = None
) -> None:
    """
    Main entry point for processing a dataset.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output CSV file
        config: Configuration dictionary (uses defaults if None)
        mode: Processing mode ('test', 'valid', or None for auto-detect)
    """
    # Merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Auto-detect mode if not specified
    if mode is None:
        basename = os.path.basename(input_file).lower()
        if 'val' in basename or 'dev' in basename:
            mode = 'valid'
        else:
            mode = 'test'
    
    # Execute appropriate mode
    if mode.startswith('val') or mode == 'valid':
        await run_validation_mode(input_file, output_file, cfg)
    else:
        await run_test_mode(input_file, output_file, cfg)


def _save_submission_file(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save submission CSV file sorted by QID.
    
    Args:
        results: List of result dictionaries
        output_file: Path to main output file (used to determine submission path)
    """
    submission_file = os.path.join(os.path.dirname(output_file), 'submission.csv')
    results_sorted = sorted(results, key=lambda x: x.get("qid", ""))
    
    with open(submission_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({"qid": row["qid"], "answer": row["answer"]})
    
    print(f"✅ Submission saved to: {submission_file}")
