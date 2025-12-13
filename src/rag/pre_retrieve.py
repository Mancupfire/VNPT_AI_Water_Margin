import os
import sys
import json
import asyncio
from tqdm.asyncio import tqdm
import numpy as np
import faiss
import pickle
from typing import Dict, Any

# Add project root to path for direct script execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.providers.factory import load_embedding_provider
from sentence_transformers import CrossEncoder
from src.logger import get_logger

logger = get_logger(__name__)

async def pre_retrieve_for_dataset(input_file: str, output_file: str, config: Dict[str, Any]):
    logger.info(f"Starting pre-retrieval for {input_file}...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return

    embedding_provider = load_embedding_provider(config)
    
    faiss_index = None
    bm25_index = None
    text_chunks = None
    cross_encoder = None

    logger.info("Loading RAG indexes...")
    try:
        with open(config["TEXT_CHUNKS_PATH"], 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
        
        if os.path.exists(config["FAISS_INDEX_PATH"]):
            faiss_index = faiss.read_index(config["FAISS_INDEX_PATH"])
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")

        if config["HYBRID_SEARCH_ENABLED"] and os.path.exists(config["BM25_INDEX_PATH"]):
            with open(config["BM25_INDEX_PATH"], 'rb') as f:
                bm25_index = pickle.load(f)
            logger.info("Loaded BM25 index.")

        if config["RERANK_ENABLED"]:
            cross_encoder = CrossEncoder(config["CROSS_ENCODER_MODEL"])
            logger.info(f"Loaded Cross-Encoder model: {config['CROSS_ENCODER_MODEL']}")
    except Exception as e:
        logger.error(f"Error loading RAG indexes: {e}. Aborting pre-retrieval.", exc_info=True)
        return

    new_data = []
    for item in tqdm(data, desc="Pre-retrieving context"):
        question_text = item.get("question", "")
        if not question_text:
            new_data.append(item)
            continue

        retrieval_top_k = config["RERANK_TOP_K"] if config["RERANK_ENABLED"] else config["TOP_K_RAG"]
        
        retrieved_chunks = []
        if config["HYBRID_SEARCH_ENABLED"] and bm25_index and faiss_index:
            tokenized_query = question_text.split(" ")
            bm25_scores = bm25_index.get_scores(tokenized_query)
            
            question_embedding_list = await embedding_provider.aembed([question_text], config)
            question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
            D, I = faiss_index.search(question_embedding, len(text_chunks))

            bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
            faiss_scores_norm = 1 - (D[0] - np.min(D[0])) / (np.max(D[0]) - np.min(D[0]))

            hybrid_scores = (config["SEMANTIC_WEIGHT"] * faiss_scores_norm) + (config["KEYWORD_WEIGHT"] * bm25_scores_norm)
            
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            retrieved_chunks = [text_chunks[i] for i in sorted_indices[:retrieval_top_k]]
        elif faiss_index:
            question_embedding_list = await embedding_provider.aembed([question_text], config)
            question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
            D, I = faiss_index.search(question_embedding, retrieval_top_k)
            retrieved_chunks = [text_chunks[idx] for idx in I[0] if idx >= 0 and idx < len(text_chunks)]

        final_context = ""
        if retrieved_chunks:
            if config["RERANK_ENABLED"] and cross_encoder:
                pairs = [[question_text, chunk] for chunk in retrieved_chunks]
                scores = cross_encoder.predict(pairs)
                sorted_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]
                final_chunks = sorted_chunks[:config["TOP_K_RAG"]]
                final_context = "\n".join(final_chunks)
            else:
                final_context = "\n".join(retrieved_chunks)

        new_item = item.copy()
        new_item["retrieved_context"] = final_context
        new_data.append(new_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Pre-retrieval complete. Output saved to {output_file}")


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    config = {
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai_hackathon_embedding"),
        "RAG_ENABLED": True, # Must be true to load indexes
        "HYBRID_SEARCH_ENABLED": os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true",
        "RERANK_ENABLED": os.getenv("RERANK_ENABLED", "false").lower() == "true",
        "FAISS_INDEX_PATH": os.getenv("FAISS_INDEX_PATH", "knowledge_base/faiss_index.bin"),
        "TEXT_CHUNKS_PATH": os.getenv("TEXT_CHUNKS_PATH", "knowledge_base/text_chunks.json"),
        "BM25_INDEX_PATH": os.getenv("BM25_INDEX_PATH", "knowledge_base/bm25_index.pkl"),
        "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "RERANK_TOP_K": int(os.getenv("RERANK_TOP_K", "10")),
        "TOP_K_RAG": int(os.getenv("TOP_K_RAG", "3")),
        "SEMANTIC_WEIGHT": float(os.getenv("SEMANTIC_WEIGHT", "0.5")),
        "KEYWORD_WEIGHT": float(os.getenv("KEYWORD_WEIGHT", "0.5")),
    }
    
    input_dataset = "data/test.json"
    output_dataset = "data/test_with_context.json"

    asyncio.run(pre_retrieve_for_dataset(input_dataset, output_dataset, config))
