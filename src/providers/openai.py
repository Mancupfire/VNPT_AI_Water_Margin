import os
from typing import Dict, Any, List

try:
    import openai
except Exception:
    openai = None

import requests


class OpenAIProvider:
    """Combined provider supporting either the OpenAI API (library) or an
    Ollama-compatible HTTP backend. Behavior is chosen by config/env:

    - If `provider` == 'ollama' or `OLLAMA_BASE` is set -> use HTTP POST to that base URL.
    - Otherwise attempt to use the `openai` Python package and `OPENAI_API_KEY`.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}
        # If an Ollama base is present, we'll use HTTP mode
        self.ollama_base = self.cfg.get("OLLAMA_BASE") or os.getenv("OLLAMA_BASE")
        # optionally override OpenAI model env var
        self.openai_model = self.cfg.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
        # configure openai key if available
        if openai is not None:
            key = self.cfg.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if key:
                openai.api_key = key

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        # If Ollama (HTTP) mode requested or base present -> call HTTP endpoint
        provider_name = (config or {}).get("PROVIDER") or os.getenv("PROVIDER")
        if provider_name and str(provider_name).lower().strip().startswith("ollama"):
            return self._call_ollama(messages, config)
        if self.ollama_base:
            return self._call_ollama(messages, config)

        # Otherwise try OpenAI Python lib
        if openai is None:
            raise RuntimeError("OpenAI package not installed; cannot use OpenAI provider")

        model = config.get("MODEL_NAME") or self.openai_model or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        # Use ChatCompletion interface
        response = openai.ChatCompletion.create(model=model, messages=messages)
        choices = response.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
        return ""

    def _call_ollama(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        # Resolve base and try a few common Ollama-compatible endpoints
        raw_base = (config or {}).get("OLLAMA_BASE") or self.ollama_base or os.getenv("OLLAMA_BASE")
        candidates = []
        if raw_base:
            candidates.append(raw_base)
            # if user provided root like http://localhost:11434, try appending paths
            if not raw_base.rstrip('/').endswith('/v1/chat/completions'):
                candidates.append(raw_base.rstrip('/') + '/v1/chat/completions')
            if not raw_base.rstrip('/').endswith('/v1/completions'):
                candidates.append(raw_base.rstrip('/') + '/v1/completions')
        else:
            # default Ollama local endpoint
            candidates = [
                'http://localhost:11434/v1/chat/completions',
                'http://localhost:11434/v1/completions'
            ]

        model = config.get("MODEL_NAME") or os.getenv("MODEL_NAME")
        payload = {"model": model, "messages": messages}

        last_err = None
        for url in candidates:
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 404:
                    # try next candidate
                    last_err = f"404 Not Found at {url}"
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    if data.get("choices"):
                        return data["choices"][0].get("message", {}).get("content", "")
                    if data.get("results"):
                        first = data["results"][0]
                        if isinstance(first, dict):
                            return first.get("content", "") or first.get("text", "")
                # if response doesn't match known shapes, return full JSON as string
                return str(data)
            except requests.exceptions.HTTPError as he:
                # For HTTP errors, capture message and continue trying other endpoints
                last_err = f"HTTP error for {url}: {he} - body: {getattr(he.response, 'text', '')}"
                continue
            except Exception as e:
                last_err = str(e)
                continue

        # If we exhausted candidates, raise a clear error
        raise RuntimeError(f"Failed to call Ollama/OpenAI HTTP endpoint. Tried: {candidates}. Last error: {last_err}")


def create(config: Dict[str, Any] | None = None) -> OpenAIProvider:
    return OpenAIProvider(config)
