from typing import Dict, Any
import os

def load_chat_provider(config: Dict[str, Any] | None = None):
    """Load a chat provider by name."""
    provider_name = (config or {}).get("CHAT_PROVIDER") or os.getenv("CHAT_PROVIDER") or "vnpt"
    name = str(provider_name).strip().lower()
    
    if name == "vnpt":
        from .vnpt import create as _create
        return _create(config)
    if name == "ollama":
        from .ollama import create as _create
        return _create(config)
    if name == "openai":
        from .openai import create as _create
        return _create(config)
    
    raise ValueError(f"Unknown chat provider: {name}")

def load_embedding_provider(config: Dict[str, Any] | None = None):
    """Load an embedding provider by name."""
    provider_name = (config or {}).get("EMBEDDING_PROVIDER") or os.getenv("EMBEDDING_PROVIDER") or "vnpt"
    name = str(provider_name).strip().lower()

    if name == "vnpt":
        from .vnpt import create as _create
        return _create(config)
    if name == "huggingface":
        from .huggingface import create as _create
        return _create(config)

    raise ValueError(f"Unknown embedding provider: {name}")

