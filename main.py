import os
import asyncio
from src.async_running import process_dataset_async

try:
    # optional: auto-load .env for local development
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _build_config_from_env():
    return {
        "VNPT_ACCESS_TOKEN": os.getenv("VNPT_ACCESS_TOKEN"),
        "VNPT_TOKEN_ID": os.getenv("VNPT_TOKEN_ID"),
        "VNPT_TOKEN_KEY": os.getenv("VNPT_TOKEN_KEY"),
        "CHAT_PROVIDER": os.getenv("CHAT_PROVIDER", "vnpt"),
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-small"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CONCURRENT_REQUESTS": int(os.getenv("CONCURRENT_REQUESTS", "2")),
        "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "90")),
        "PAYLOAD_HYPERPARAMS": {
            "temperature": 0.5,
            "top_p": 0.7,
            "max_completion_tokens": 2048,
            "n": 1,
            "seed": 416,
        },
        "RAG_ENABLED": os.getenv("RAG_ENABLED", "false").lower() == "true",
        "TOP_K_RAG": int(os.getenv("TOP_K_RAG", "3")),
        "FAISS_INDEX_PATH": os.getenv("FAISS_INDEX_PATH", "knowledge_base/faiss_index.bin"),
        "TEXT_CHUNKS_PATH": os.getenv("TEXT_CHUNKS_PATH", "knowledge_base/text_chunks.json"),
        "RERANK_ENABLED": os.getenv("RERANK_ENABLED", "false").lower() == "true",
        "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "RERANK_TOP_K": int(os.getenv("RERANK_TOP_K", "10")),
        "HYBRID_SEARCH_ENABLED": os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true",
        "BM25_INDEX_PATH": os.getenv("BM25_INDEX_PATH", "knowledge_base/bm25_index.pkl"),
        "SEMANTIC_WEIGHT": float(os.getenv("SEMANTIC_WEIGHT", "0.5")),
        "KEYWORD_WEIGHT": float(os.getenv("KEYWORD_WEIGHT", "0.5")),
    }


if __name__ == "__main__":
    config = _build_config_from_env()
    
    # It's recommended to run one process at a time
    # to avoid race conditions on the output files.
    
    # Example for test set
    asyncio.run(
        process_dataset_async(
            input_file='E:\\VNPT_AI_Water_Margin\\data\\test.json',
            output_file=f'pred/test_{config.get("CHAT_PROVIDER", "")}_async.csv',
            config=config,
            mode='test'
        )
    )

    # Example for validation set
    # asyncio.run(
    #     process_dataset_async(
    #         input_file='E:\\VNPT_AI_Water_Margin\\data\\val.json',
    #         output_file=f'pred/val_{config.get("CHAT_PROVIDER", "")}_async.csv',
    #         config=config,
    #         mode='valid'
    #     )
    # )
