"""
Backwards compatibility wrapper for async_running.py

DEPRECATED: This file is kept for backwards compatibility.
New code should import from:
- src.core.runner for main orchestration
- src.core.processor for item processing
- src.utils for utility functions

The original 405-line monolithic file has been refactored into:
- src/utils/ - Prompt formatting, prediction cleaning, progress management
- src/RAG/ - Context retrieval and index loading
- src/core/ - Core processing logic and orchestration
"""

# Import from new modular structure
from src.core.config import DEFAULT_CONFIG
from src.core.processor import process_item
from src.core.runner import (
    process_dataset as process_dataset_async,
    run_test_mode as test_function_async,
    run_validation_mode as valid_function_async
)
from src.utils.prompt import format_prompt
from src.utils.prediction import clean_prediction

# Re-export for backwards compatibility
__all__ = [
    "DEFAULT_CONFIG",
    "format_prompt",
    "clean_prediction",
    "process_item",
    "test_function_async",
    "valid_function_async",
    "process_dataset_async",
]
