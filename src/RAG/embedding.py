import numpy as np
from ..language_models.call_embedding import get_embedding

def chunk_and_embed(documents, chunk_size=2048, overlap=256):
    """
    Takes a list of documents, chunks them, and embeds each chunk.
    """
    embeddings = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size, overlap)
        doc_embeddings = [get_embedding(chunk) for chunk in chunks]
        embeddings.extend(doc_embeddings)
    return embeddings

def chunk_text(text, chunk_size=2048, overlap=256):
    """
    Breaks a large document into overlapping chunks.
    """
    chunks = []
    text_len = len(text)
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move start point with overlap
    return chunks

def embed_query(query):
    """
    Embeds a single query string and returns the embedding.
    """
    return get_embedding(query)
