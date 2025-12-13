import os
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddingProvider:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}
        model_name = self.cfg.get("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    async def aembed(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [e.tolist() for e in embeddings]


def create(config: Dict[str, Any] | None = None) -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(config)
