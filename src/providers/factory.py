from typing import Dict, Any
import os

def load_provider(provider_name: str | None = None, config: Dict[str, Any] | None = None):
    """Load a provider adapter by name. Known names: 'vnpt', 'ollama', 'openai'.

    Returns a provider instance with a `chat(messages, config)` method.
    """
    raw = provider_name or (config or {}).get("PROVIDER") or os.getenv("PROVIDER") or "vnpt"
    # sanitize: strip whitespace and remove inline shell-style comments
    name = str(raw).split('#', 1)[0].strip().split()[0].lower()
    if name == "vnpt":
        from .vnpt import create as _create
        return _create(config)
    if name in ("ollama", "openai"):
        # use the combined OpenAI provider which supports both the OpenAI
        # API (via `openai` package) and Ollama-compatible HTTP backends.
        from .openai import create as _create
        return _create(config)
    raise ValueError(f"Unknown provider: {name}")
