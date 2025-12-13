import os
import sys
import json
import asyncio
from pypdf import PdfReader
import faiss
import numpy as np
from typing import Dict, Any, List
from rank_bm25 import BM25Okapi
import pickle

# Add project root to path for direct script execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.providers.factory import load_embedding_provider
from src.logger import get_logger

logger = get_logger(__name__)

# Configuration for the embedding model (from the PDF doc)
EMBEDDING_MODEL_NAME = "vnptai_hackathon_embedding"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50 # characters
EMBEDDING_DIM = 768 # Common embedding dimension, confirm with VNPT API doc if necessary

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str] = None) -> List[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    final_chunks = []
    
    # Start with the largest separator
    separator = separators[0]
    splits = text.split(separator)
    
    good_splits = []
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if len(good_splits) > 0:
                merged_text = separator.join(good_splits)
                final_chunks.extend(_recursive_chunk(merged_text, chunk_size, chunk_overlap, separators))
                good_splits = []
            
            if len(s) > chunk_size:
                # Recurse on the large split
                other_chunks = chunk_text(s, chunk_size, chunk_overlap, separators[1:])
                final_chunks.extend(other_chunks)
            else:
                final_chunks.append(s)
    
    if len(good_splits) > 0:
        merged_text = separator.join(good_splits)
        final_chunks.extend(_recursive_chunk(merged_text, chunk_size, chunk_overlap, separators))
        
    return [c for c in final_chunks if c.strip()]

def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    final_chunks = []
    if len(text) <= chunk_size:
        final_chunks.append(text)
    else:
        start_index = 0
        while start_index < len(text):
            end_index = start_index + chunk_size
            if end_index > len(text):
                end_index = len(text)
            
            chunk = text[start_index:end_index]
            final_chunks.append(chunk)
            start_index += chunk_size - chunk_overlap
            
    return final_chunks

async def generate_embeddings_async(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    embedding_provider = load_embedding_provider(config)
    try:
        embeddings = await embedding_provider.aembed(texts, config)
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM)) # Return zeros for failed embedding


async def build_index(docs_dir: str, index_path: str, texts_path: str, bm25_index_path: str, config: Dict[str, Any]):
    all_chunks = []
    logger.info(f"Loading PDFs from {docs_dir}...")
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                all_chunks.extend(chunks)
                logger.info(f"Processed {filename}: {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

    if not all_chunks:
        logger.warning("No text chunks extracted from PDFs. Index will be empty.")
        return

    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info("Generating embeddings (this may take a while)...")

    embeddings = await generate_embeddings_async(all_chunks, config)

    d = embeddings.shape[1] if embeddings.shape[0] > 0 else EMBEDDING_DIM
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True) # Create directory if it doesn't exist
    faiss.write_index(index, index_path)
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    logger.info(f"FAISS index built and saved to {index_path}")
    logger.info(f"Text chunks saved to {texts_path}")

    # Build and save BM25 index
    logger.info("Building BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_index_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index built and saved to {bm25_index_path}")


if __name__ == "__main__":
    logger.info("Starting index building process...")
    import dotenv
    dotenv.load_dotenv() # Load .env variables

    config = {
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai_hackathon_embedding"),
    }

    docs_directory = "E:\\VNPT_AI_Water_Margin\\docs"
    knowledge_base_dir = "knowledge_base"
    os.makedirs(knowledge_base_dir, exist_ok=True) # Ensure knowledge_base directory exists

    faiss_index_file = os.path.join(knowledge_base_dir, "faiss_index.bin")
    text_chunks_file = os.path.join(knowledge_base_dir, "text_chunks.json")
    bm25_index_file = os.path.join(knowledge_base_dir, "bm25_index.pkl")

    asyncio.run(build_index(docs_directory, faiss_index_file, text_chunks_file, bm25_index_file, config))
    logger.info("Index building process finished.")
