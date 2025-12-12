import json
from pathlib import Path
import re
import requests
from typing import Optional
import time
from .logging import log_to_file

# locate secrets file relative to repository root
secrets_path = Path(__file__).resolve().parents[2] / "secrets" / "api-keys.json"

def load_credentials(llm_api_name: str | None = None, token_id: str | None = None):
    with open(secrets_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for e in entries:
        if llm_api_name and e.get("llmApiName") == llm_api_name:
            return e
        if token_id and e.get("tokenId") == token_id:
            return e
    return entries[0] if entries else None

class VNPTClient:
    """
    Clean VNPT AI chat client with proper error handling.
    """

    def __init__(
        self,
        model: str = "vnptai_hackathon_small",
        token_id: str | None = None,
        timeout: int = 30,
    ):
        if model == "vnptai_hackathon_small":
            llm_api_name = "LLM small"
            self.url = ("https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small")
        else:
            llm_api_name = "LLM large"
            self.url = ("https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large")
        
        creds = load_credentials(llm_api_name=llm_api_name, token_id=token_id)

        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": creds["authorization"],
            "Token-id": creds["tokenId"],
            "Token-key": creds["tokenKey"],
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        """
        Generate text using VNPT AI ChatCompletion API.

        Args:
            prompt: User prompt
            model: Model name (VNPT model id). Defaults to the one set in __init__.
            temperature: Sampling temperature
            num_ctx: Mapped to `max_completion_tokens` in VNPT request
            system: Optional system prompt

        Returns:
            Generated text (string)
        """

        # Build OpenAI-style messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": 1.0,
            "top_k": 20,
            "n": 1,
            "max_completion_tokens": int(num_ctx),
        }

        try:
            resp = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception(f"VNPT API timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to VNPT AI endpoint")
        except requests.exceptions.HTTPError as e:
            try:
                detail = resp.text
            except Exception:
                detail = "<no body>"
            raise Exception(f"VNPT API error: {e} | body={detail}")

        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from VNPT API")

        # VNPT response is expected to be OpenAI-like:
        # { "choices": [ { "message": { "content": "..." }, ... } ], ... }
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise Exception(f"Unexpected VNPT response format: {data}")

        return content.strip()

def remove_reasoning(response_content: str) -> str:
    """Remove reasoning part if present in <think>...</think> tags."""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_content.strip()

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "vnptai_hackathon_small",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    num_retries: int = 3,
) -> str:
    """
    Args:
        user_prompt: User prompt
        sys_prompt: System prompt (optional)
        model_name: VNPT model name (default: "vnptai_hackathon_small")
        max_tokens: Mapped to `max_completion_tokens` in VNPT request
        temperature: Sampling temperature
        infinite_retry: If True, retry forever on any exception

    Returns:
        Generated text (with any <think> block stripped out).
    """

    vnpt_client = VNPTClient(model=model_name)
    
    while num_retries > 0:
        num_retries -= 1
        try:
            response = vnpt_client.generate(
                prompt=user_prompt,
                model=model_name,
                temperature=temperature,
                num_ctx=max_tokens,
                system=sys_prompt if sys_prompt else None,
            )
            return remove_reasoning(response)
        except Exception as e:
            log_to_file(filename="llm_excepts", message=f"str({e})")
            time.sleep(30)
            continue
    
    log_to_file(filename="llm_errors", message=f"ask() failed after retries: {user_prompt}")
    
    return "C"

if __name__ == "__main__":
    # Simple test
    prompt = "Viết một đoạn văn ngắn về lợi ích của việc đọc sách."
    response = ask(model_name="vnptai_hackathon_large", user_prompt=prompt, num_retries=2)
    print("Response:", response)