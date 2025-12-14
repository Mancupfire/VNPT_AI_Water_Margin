import os
import json
import asyncio
from tqdm.asyncio import tqdm
from typing import Dict, Any, List
import csv
import faiss
import numpy as np
import pickle
from pyvi import ViTokenizer

from src.providers import load_chat_provider
from sentence_transformers import CrossEncoder

DEFAULT_CONFIG: Dict[str, Any] = {
    "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-small"),
    "CONCURRENT_REQUESTS": int(os.getenv("CONCURRENT_REQUESTS", "2")),
    "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "90")),
    "PROVIDER": os.getenv("PROVIDER", "vnpt"),
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

def format_prompt(item, context: str = None):
    question = item.get('question', '')
    choices = item.get('choices', [])

    labels = [chr(ord('A') + i) for i in range(26)]
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            formatted_choices.append(f"{labels[i]}. {choice}")

    prompt_text = (
        f"{question}" + "\n".join(formatted_choices)
    )
    
    if context:
        prompt_text = f"Context: {context}\n\nQuestion: {prompt_text}"

    return [
        {
            "role": "system",
            "content": "Bạn là 1 trợ lý ảo AI thông minh, cẩn thận và chính xác. Với tư cách là 1 chuyên gia người Việt Nam, yêu nước, hòa đồng, thân thiện, nhiệm vụ của bạn là hãy suy nghĩ trả lời từng bước một sau đó đưa ra câu trả lời cho câu hỏi trắc nhiệm sau đây bằng cách đưa ra ký tự chữ cái in hoa đại diện cho câu trả lời đó theo định dạng 'Vậy đáp án là X' trong đó X là chữ cái đại diện cho câu trả lời đúng nhất. \n Nếu câu hỏi đó mang ý đồ xấu, vi phạm pháp luật, đạo đức bạn cần chọn đáp án không trả lời trừ khi phục vụ mục đích giáo dục."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]


def _clean_prediction(pred_text):
    DEFAULT_ANSWER = "C"
    if not pred_text:
        return DEFAULT_ANSWER
    pred_text = str(pred_text).strip()
    if "Đáp án:" in pred_text:
        parts = pred_text.split("Đáp án:", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    if "Đáp án đúng:" in pred_text:
        parts = pred_text.split("Đáp án đúng:", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    if "Vậy đáp án là" in pred_text:
        parts = pred_text.split("Vậy đáp án là", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    if pred_text[-3] .isalpha() or pred_text[-4] .isalpha():
        return pred_text[-3].upper() if pred_text[-3] .isalpha() else pred_text[-4].upper()
    return DEFAULT_ANSWER

async def call_llm_async(provider, messages: list[Dict[str, Any]], config: Dict[str, Any] | None = None) -> str:
    """Dispatch async chat request to provider and return response text."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    return await provider.achat(messages, cfg)

async def process_item(item, provider, config: Dict[str, Any], semaphore: asyncio.Semaphore, faiss_index=None, bm25_index=None, text_chunks=None, cross_encoder=None):
    async with semaphore:
        context = None
        if config.get("RAG_ENABLED") and text_chunks is not None:
            question_text = item.get("question", "")
            if question_text:
                retrieval_top_k = config.get("RERANK_TOP_K", 10) if config.get("RERANK_ENABLED") else config.get("TOP_K_RAG", 3)
                
                retrieved_chunks = []
                if config.get("HYBRID_SEARCH_ENABLED") and bm25_index is not None and faiss_index is not None:
                    # Hybrid Search
                    tokenized_query = ViTokenizer.tokenize(question_text).split()
                    
                    loop = asyncio.get_running_loop()
                    bm25_scores = await loop.run_in_executor(None, bm25_index.get_scores, tokenized_query)
                    
                    question_embedding_list = await provider.aembed([question_text], {"MODEL_NAME": "vnptai_hackathon_embedding"})
                    question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
                    
                    # Use run_in_executor for CPU-bound FAISS search
                    D, I = await loop.run_in_executor(None, lambda: faiss_index.search(question_embedding, len(text_chunks)))

                    # Normalize scores
                    bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
                    if bm25_max - bm25_min == 0:
                        bm25_scores_norm = np.zeros_like(bm25_scores)
                    else:
                        bm25_scores_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)

                    faiss_min, faiss_max = np.min(D[0]), np.max(D[0])
                    if faiss_max - faiss_min == 0:
                        faiss_scores_norm = np.zeros_like(D[0])
                    else:
                        faiss_scores_norm = 1 - (D[0] - faiss_min) / (faiss_max - faiss_min)

                    # Combine scores
                    hybrid_scores = (config["SEMANTIC_WEIGHT"] * faiss_scores_norm) + (config["KEYWORD_WEIGHT"] * bm25_scores_norm)
                    
                    sorted_indices = np.argsort(hybrid_scores)[::-1]
                    retrieved_chunks = [text_chunks[i] for i in sorted_indices[:retrieval_top_k]]
                elif faiss_index is not None:
                    # Dense Search only
                    question_embedding_list = await provider.aembed([question_text], {"MODEL_NAME": "vnptai_hackathon_embedding"})
                    question_embedding = np.array(question_embedding_list[0]).reshape(1, -1)
                    
                    loop = asyncio.get_running_loop()
                    # Use run_in_executor for CPU-bound FAISS search
                    D, I = await loop.run_in_executor(None, lambda: faiss_index.search(question_embedding, retrieval_top_k))
                    retrieved_chunks = [text_chunks[idx] for idx in I[0] if idx >= 0 and idx < len(text_chunks)]

                # Re-ranking
                if config.get("RERANK_ENABLED") and cross_encoder is not None:
                    pairs = [[question_text, chunk] for chunk in retrieved_chunks]
                    scores = cross_encoder.predict(pairs)
                    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]
                    final_chunks = sorted_chunks[:config.get("TOP_K_RAG", 3)]
                    context = "\n".join(final_chunks)
                else:
                    context = "\n".join(retrieved_chunks)

        messages = format_prompt(item, context)
        prediction_text = await call_llm_async(provider, messages, config)
        
        sleep_time = config.get("SLEEP_TIME", DEFAULT_CONFIG['SLEEP_TIME'])
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        if not prediction_text:
            clean_prediction = "C"
        elif len(prediction_text) == 1 and prediction_text.isalpha() and prediction_text.isupper():
            clean_prediction = prediction_text
        elif prediction_text.startswith("Error from VNPT API:"):
            for i, choice in enumerate(item.get('choices', [])):
                if any(phrase in choice for phrase in ["Không thể", "Xin lỗi", "Tôi không thể trả lời", "Không trả lời"]):
                    clean_prediction = chr(ord('A') + i)
                    break
        else:
            clean_prediction = _clean_prediction(prediction_text)

        result = {
            "qid": item.get('qid'),
            "answer": clean_prediction,
            "prediction_raw": prediction_text.replace('\n', ' ')
        }
        if 'answer' in item:
            ground_truth = item.get('answer', '')
            is_correct = (clean_prediction.upper() == ground_truth.upper()) if ground_truth else False
            result["ground_truth"] = ground_truth
            result["is_correct"] = is_correct
        return result


async def test_function_async(input_file, output_csv, config=None):
    print(f"--- Processing (async test): {input_file} -> {output_csv} ---")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    provider = load_chat_provider(config)
    semaphore = asyncio.Semaphore(config.get("CONCURRENT_REQUESTS", 2))

    faiss_index = None
    bm25_index = None
    text_chunks = None
    cross_encoder = None
    if config.get("RAG_ENABLED"):
        try:
            with open(config.get("TEXT_CHUNKS_PATH"), 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            
            if os.path.exists(config.get("FAISS_INDEX_PATH")):
                faiss_index = faiss.read_index(config.get("FAISS_INDEX_PATH"))
                print(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")

            if config.get("HYBRID_SEARCH_ENABLED") and os.path.exists(config.get("BM25_INDEX_PATH")):
                with open(config.get("BM25_INDEX_PATH"), 'rb') as f:
                    bm25_index = pickle.load(f)
                print("Loaded BM25 index.")

            if config.get("RERANK_ENABLED"):
                cross_encoder = CrossEncoder(config.get("CROSS_ENCODER_MODEL"))
                print(f"Loaded Cross-Encoder model: {config.get('CROSS_ENCODER_MODEL')}")
        except Exception as e:
            print(f"Error loading RAG indexes or Cross-Encoder: {e}. RAG will be disabled.")
            config["RAG_ENABLED"] = False

    # === CHECKPOINT/RESUME LOGIC ===
    processed_qids = set()
    existing_results = []
    
    if os.path.exists(output_csv):
        print(f"\n📋 Found existing progress file: {output_csv}")
        try:
            with open(output_csv, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_qids.add(row['qid'])
                    existing_results.append(row)
            print(f"✅ Loaded {len(processed_qids)} already processed items")
        except Exception as e:
            print(f"⚠️  Error reading existing file: {e}. Starting fresh...")
    else:
        print(f"\n📄 No existing progress file. Starting fresh...")
    
    # Filter out already processed items
    items_to_process = [item for item in data if item['qid'] not in processed_qids]
    
    if not items_to_process:
        print(f"\n✅ All {len(data)} items already processed! Nothing to do.")
        return
    
    print(f"📊 Progress: {len(processed_qids)}/{len(data)} complete. Processing {len(items_to_process)} remaining items...\n")

    tasks = [process_item(item, provider, config, semaphore, faiss_index, bm25_index, text_chunks, cross_encoder) for item in items_to_process]
    
    # Incremental saving: write results as they complete
    results = existing_results.copy()
    fieldnames = ["qid", "answer", "prediction_raw"]
    
    
    # Open in APPEND mode to preserve existing content
    with open(output_csv, 'a', encoding='utf-8', newline='') as f_result:
        writer_result = csv.DictWriter(f_result, fieldnames=fieldnames)
        
        # Only write header if file is new/empty
        if not existing_results:
            writer_result.writeheader()
            f_result.flush()
        
        with tqdm(total=len(items_to_process), desc="Processing") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Write to result CSV immediately
                writer_result.writerow(result)
                f_result.flush()
                
                pbar.update(1)

    print(f"Done. Results saved to: {output_csv}")
    
    # Sort by qid and save submission.csv
    submission_file = os.path.join(os.path.dirname(output_csv), 'submission.csv')
    results_sorted = sorted(results, key=lambda x: x.get("qid", ""))
    with open(submission_file, 'w', encoding='utf-8', newline='') as f_submission:
        writer_submission = csv.DictWriter(f_submission, fieldnames=["qid", "answer"])
        writer_submission.writeheader()
        for row in results_sorted:
            writer_submission.writerow({"qid": row["qid"], "answer": row["answer"]})
    print(f"Submission saved to: {submission_file}")


async def valid_function_async(input_file, output_csv, config=None):
    print(f"--- Processing (async validation): {input_file} -> {output_csv} ---")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    provider = load_chat_provider(config)
    semaphore = asyncio.Semaphore(config.get("CONCURRENT_REQUESTS", 2))

    faiss_index = None
    bm25_index = None
    text_chunks = None
    cross_encoder = None
    if config.get("RAG_ENABLED"):
        try:
            with open(config.get("TEXT_CHUNKS_PATH"), 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)

            if os.path.exists(config.get("FAISS_INDEX_PATH")):
                faiss_index = faiss.read_index(config.get("FAISS_INDEX_PATH"))
                print(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")

            if config.get("HYBRID_SEARCH_ENABLED") and os.path.exists(config.get("BM25_INDEX_PATH")):
                with open(config.get("BM25_INDEX_PATH"), 'rb') as f:
                    bm25_index = pickle.load(f)
                print("Loaded BM25 index.")

            if config.get("RERANK_ENABLED"):
                cross_encoder = CrossEncoder(config.get("CROSS_ENCODER_MODEL"))
                print(f"Loaded Cross-Encoder model: {config.get('CROSS_ENCODER_MODEL')}")
        except Exception as e:
            print(f"Error loading RAG indexes or Cross-Encoder: {e}. RAG will be disabled.")
            config["RAG_ENABLED"] = False

    tasks = [process_item(item, provider, config, semaphore, faiss_index, bm25_index, text_chunks, cross_encoder) for item in data]
    
    # Incremental saving: write results as they complete
    results = []
    correct = 0
    total = len(tasks)
    fieldnames = ["qid", "prediction", "ground_truth", "is_correct", "prediction_raw"]
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        
        with tqdm(total=total, desc="Processing") as pbar:
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                
                if res.get('is_correct'):
                    correct += 1
                
                # Write to result CSV immediately
                writer.writerow({
                    "qid": res.get("qid"),
                    "prediction": res.get("answer"),
                    "ground_truth": res.get("ground_truth"),
                    "is_correct": res.get("is_correct"),
                    "prediction_raw": res.get("prediction_raw")
                })
                f.flush()
                
                # Update progress bar with current accuracy
                current_acc = correct / len(results) if results else 0
                pbar.set_postfix({"acc": f"{current_acc:.2%}", "correct": correct})
                pbar.update(1)

    acc = correct / total if total > 0 else 0.0
    print(f"Done. Results saved to: {output_csv}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    
    # Sort by qid and save submission.csv
    submission_file = os.path.join(os.path.dirname(output_csv), 'submission.csv')
    results_sorted = sorted(results, key=lambda x: x.get("qid", ""))
    with open(submission_file, 'w', encoding='utf-8', newline='') as f_submission:
        writer_submission = csv.DictWriter(f_submission, fieldnames=["qid", "answer"])
        writer_submission.writeheader()
        for row in results_sorted:
            writer_submission.writerow({"qid": row["qid"], "answer": row["answer"]})
    print(f"Submission saved to: {submission_file}")


async def process_dataset_async(input_file, output_file, config=None, mode=None):
    if mode is None:
        bname = os.path.basename(input_file).lower()
        if 'val' in bname or 'dev' in bname:
            mode = 'valid'
        else:
            mode = 'test'

    if mode.startswith('val') or mode == 'valid':
        await valid_function_async(input_file, output_file, config=config)
    else:
        await test_function_async(input_file, output_file, config=config)
