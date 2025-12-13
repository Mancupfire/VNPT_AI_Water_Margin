import os
from typing import Dict, Any, List
import asyncio
try:
    import openai
    from openai import AsyncOpenAI
except Exception:
    openai = None


class OpenAIProvider:
    """Provider for the OpenAI API."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}
        self.openai_model = self.cfg.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
        if openai is not None:
            key = self.cfg.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if key:
                self.async_client = AsyncOpenAI(api_key=key)
                openai.api_key = key

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.achat(messages, config))

    async def achat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        if openai is None:
            raise RuntimeError("OpenAI package not installed; cannot use OpenAI provider")

        model = config.get("MODEL_NAME") or self.openai_model or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        response = await self.async_client.chat.completions.create(model=model, messages=messages)
        choices = response.choices
        if choices:
            return choices[0].message.content
        return ""


def create(config: Dict[str, Any] | None = None) -> OpenAIProvider:
    return OpenAIProvider(config)
