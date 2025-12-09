import os
import requests
from .base import Provider
from typing import Dict, Any, List


class VNPTProvider:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VNPT_API_BASE", "https://api.idg.vnpt.vn/data-service/v1/chat/completions")

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        model_name = config.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "vnptai-hackathon-small"
        api_base = self.base_url.rstrip('/')
        # The VNPT API expects the model in the request body and the endpoint
        # to be the chat/completions path. Do not append the model to the URL.
        url = f"{api_base}/{model_name}"

        headers = {"Content-Type": "application/json"}
        access_token = config.get("ACCESS_TOKEN") or os.getenv("VNPT_ACCESS_TOKEN")
        token_id = config.get("TOKEN_ID") or os.getenv("VNPT_TOKEN_ID")
        token_key = config.get("TOKEN_KEY") or os.getenv("VNPT_TOKEN_KEY")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if token_id:
            headers["Token-id"] = token_id
        if token_key:
            headers["Token-key"] = token_key

        # Some VNPT deployments expect model names with underscores in the body
        model_body_name = model_name.replace('-', '_')
        payload = {"model": model_body_name, "messages": messages}
        payload.update(config.get("PAYLOAD_HYPERPARAMS", {}))

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data:
            return f"Error from VNPT API: {data['error'].get('message', str(data['error']))}"
        if isinstance(data, dict) and data.get("choices"):
            return data["choices"][0]["message"]["content"]
        return ""


def create(config: Dict[str, Any] | None = None) -> VNPTProvider:
    return VNPTProvider()
