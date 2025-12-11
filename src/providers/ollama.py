import os
from typing import Dict, Any, List
import asyncio
import aiohttp


class OllamaProvider:
    """Provider for Ollama-compatible backends."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}
        self.ollama_base = self.cfg.get("OLLAMA_BASE") or os.getenv("OLLAMA_BASE", 'http://localhost:11434')

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.achat(messages, config))

    async def achat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        raw_base = (config or {}).get("OLLAMA_BASE") or self.ollama_base
        url = f"{raw_base.rstrip('/')}/api/chat"

        model = config.get("MODEL_NAME") or os.getenv("MODEL_NAME")
        payload = {
            "model": model, 
            "messages": messages,
            "stream": False
        }
        payload.update(config.get("PAYLOAD_HYPERPARAMS", {}))

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if isinstance(data, dict) and "message" in data:
                    return data["message"]["content"]
                return str(data)


def create(config: Dict[str, Any] | None = None) -> OllamaProvider:
    return OllamaProvider(config)
