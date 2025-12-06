import gc
import torch

def cleanup():
    """Release unreferenced GPU memory and clear cache."""
    gc.collect()  # Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # Clears cache for reuse
        torch.cuda.ipc_collect()   # Cleans up interprocess memory
