import json
from pathlib import Path
import requests
from typing import Iterable, List, Union
from .logging import log_to_file

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

VNPT_EMBED_URL = ("https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding")

class VNPTEmbeddingClient:
    """
    VNPT AI Embedding client.

    Expected to be OpenAI-compatible:
    request:  { "model": "...", "input": "text" or ["text1", "text2", ...] }
    response: { "data": [ {"embedding": [...], "index": 0}, ... ], ... }
    """

    def __init__(
        self,
        model: str = "vnptai_hackathon_embedding",
        llm_api_name: str = "Embedding large",  # match name in api-keys.json
        token_id: str | None = None,
        timeout: int = 30,
    ):
        creds = load_credentials(llm_api_name=llm_api_name, token_id=token_id)
        if not creds:
            raise RuntimeError(f"No API keys found in {secrets_path}")

        self.model = model
        self.timeout = timeout
        self.url = VNPT_EMBED_URL
        self.headers = {
            "Authorization": creds["authorization"],
            "Token-id": creds["tokenId"],
            "Token-key": creds["tokenKey"],
            "Content-Type": "application/json",
        }

    def embed(
        self,
        inputs: Union[str, Iterable[str]],
        model: str | None = None,
    ) -> List[List[float]]:
        """
        Get embeddings for one or many texts.

        Args:
            inputs: A single string or an iterable of strings.
            model:  VNPT embedding model id (optional override).

        Returns:
            List of embedding vectors (one per input).
        """
        # Normalize to list of strings
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = list(inputs)

        payload = {
            "model": model or self.model,
            "input": input_list,
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
            raise Exception(f"VNPT Embedding API timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to VNPT Embedding endpoint")
        except requests.exceptions.HTTPError as e:
            try:
                detail = resp.text
            except Exception:
                detail = "<no body>"
            raise Exception(f"VNPT Embedding API error: {e} | body={detail}")

        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from VNPT Embedding API")

        # OpenAI-like: { "data": [ { "embedding": [...], ... }, ... ] }
        try:
            embeddings = [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError):
            raise Exception(f"Unexpected VNPT Embedding response format: {data}")

        return embeddings

vnpt_embedding_client = VNPTEmbeddingClient()

def get_embedding(
    text: str,
    model_name: str = "vnptai_hackathon_embedding",
    infinite_retry: bool = False,
) -> List[float]:
    """
    Get embedding for a single text.
    Returns a single embedding vector.
    """

    def _once() -> List[float]:
        embeddings = vnpt_embedding_client.embed(
            inputs=[text],
            model=model_name,
        )
        return embeddings[0]

    if infinite_retry:
        while True:
            try:
                return _once()
            except Exception as e:
                continue

    return _once()

def get_embeddings(
    texts: Iterable[str],
    model_name: str = "vnptai_hackathon_embedding",
    infinite_retry: bool = False,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts.
    Returns a list of embedding vectors.
    """

    def _once() -> List[List[float]]:
        return vnpt_embedding_client.embed(
            inputs=texts,
            model=model_name,
        )

    if infinite_retry:
        while True:
            try:
                return _once()
            except Exception as e:
                continue

    return _once()

if __name__ == "__main__":
    sample_text = "Việc đọc sách giúp mở rộng tri thức và phát triển tư duy."
    emb = get_embedding(sample_text)
    print("Embedding dim:", len(emb))
    print("Embedding vector (first 5 values):", emb[:5])
