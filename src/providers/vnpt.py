import os
import requests
import aiohttp
from .base import Provider
from typing import Dict, Any, List
import asyncio


class VNPTProvider:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VNPT_API_BASE", "https://api.idg.vnpt.vn/data-service/v1/chat/completions")
        self.embedding_url = os.getenv("VNPT_EMBEDDING_API_BASE", "https://api.idg.vnpt.vn/data-service/v1/vnptai-hackathon-embedding")

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.achat(messages, config))

    def _get_credentials(self, model_name: str, config: Dict[str, Any]):
        """Helper to get the correct Token ID and Key based on the model name."""
        access_token = config.get("ACCESS_TOKEN") or os.getenv("VNPT_ACCESS_TOKEN")
        
        token_id = None
        token_key = None

        if "small" in model_name.lower():
            token_id = os.getenv("VNPT_SMALL_TOKEN_ID")
            token_key = os.getenv("VNPT_SMALL_TOKEN_KEY")
        elif "large" in model_name.lower():
            token_id = os.getenv("VNPT_LARGE_TOKEN_ID")
            token_key = os.getenv("VNPT_LARGE_TOKEN_KEY")
        elif "embedding" in model_name.lower():
            token_id = os.getenv("VNPT_EMBEDDING_TOKEN_ID")
            token_key = os.getenv("VNPT_EMBEDDING_TOKEN_KEY")
        
        # Fallback to generic if specific ones are missing
        if not token_id:
            token_id = config.get("TOKEN_ID") or os.getenv("VNPT_TOKEN_ID")
        if not token_key:
            token_key = config.get("TOKEN_KEY") or os.getenv("VNPT_TOKEN_KEY")
            
        return access_token, token_id, token_key

    async def achat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        model_name = config.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "vnptai-hackathon-small"
        api_base = self.base_url.rstrip('/')
        url = f"{api_base}/{model_name}"

        headers = {"Content-Type": "application/json"}
        access_token, token_id, token_key = self._get_credentials(model_name, config)

        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if token_id:
            headers["Token-id"] = token_id
        if token_key:
            headers["Token-key"] = token_key

        model_body_name = model_name.replace('-', '_')
        payload = {"model": model_body_name, "messages": messages}
        payload.update(config.get("PAYLOAD_HYPERPARAMS", {}))

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if 'error' in data:
                    return f"Error from VNPT API: {data['error'].get('message', str(data['error']))}"
                if isinstance(data, dict) and data.get("choices"):
                    return data["choices"][0]["message"]["content"]
                return ""

    async def aembed(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        model_name = config.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "vnptai_hackathon_embedding"
        url = self.embedding_url

        headers = {"Content-Type": "application/json"}
        access_token, token_id, token_key = self._get_credentials(model_name, config)

        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if token_id:
            headers["Token-id"] = token_id
        if token_key:
            headers["Token-key"] = token_key

        all_embeddings = []
        for text in texts:
            payload = {
                "model": model_name,
                "input": text,
                "encoding_format": "float"
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if 'error' in data:
                        raise Exception(f"Error from VNPT Embedding API: {data['error'].get('message', str(data['error']))}")
                    if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                        all_embeddings.append(data["data"][0]["embedding"])
                    else:
                        all_embeddings.append([])
        return all_embeddings


def create(config: Dict[str, Any] | None = None) -> VNPTProvider:
    return VNPTProvider()
