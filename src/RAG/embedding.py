from langchain.embeddings.base import Embeddings
from typing import List
from ..language_models.call_embedding import get_embedding, get_embeddings

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)
