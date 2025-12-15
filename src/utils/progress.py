"""Progress tracking and checkpoint/resume utilities."""

import os
import csv
from typing import Dict, List, Set, Tuple, Any


def load_progress(output_file: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Load progress from existing CSV output file.
    
    Args:
        output_file: Path to the CSV output file
        
    Returns:
        Tuple of (processed_qids, existing_results)
        - processed_qids: Set of question IDs that have been processed
        - existing_results: List of result dictionaries
    """
    processed_qids = set()
    existing_results = []
    
    if os.path.exists(output_file):
        print(f"\n📋 Found existing progress file: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_qids.add(row['qid'])
                    existing_results.append(row)
            print(f"✅ Loaded {len(processed_qids)} already processed items")
        except Exception as e:
            print(f"⚠️  Error reading existing file: {e}. Starting fresh...")
            processed_qids = set()
            existing_results = []
    else:
        print(f"\n📄 No existing progress file. Starting fresh...")
    
    return processed_qids, existing_results


def filter_items(data: List[Dict[str, Any]], processed_qids: Set[str]) -> List[Dict[str, Any]]:
    """
    Filter out already processed items from the dataset.
    
    Args:
        data: Full dataset of question items
        processed_qids: Set of question IDs that have been processed
        
    Returns:
        List of items that haven't been processed yet
    """
    return [item for item in data if item['qid'] not in processed_qids]


def display_progress_info(processed_count: int, total_count: int, remaining_count: int) -> None:
    """
    Display progress information to the console.
    
    Args:
        processed_count: Number of items already processed
        total_count: Total number of items in dataset
        remaining_count: Number of items remaining to process
    """
    if remaining_count == 0:
        print(f"\n✅ All {total_count} items already processed! Nothing to do.")
    else:
        print(f"📊 Progress: {processed_count}/{total_count} complete. "
              f"Processing {remaining_count} remaining items...\n")
