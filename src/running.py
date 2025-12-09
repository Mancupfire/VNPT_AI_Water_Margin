import os
import json
import time
from tqdm import tqdm
from typing import Dict, Any

from src.providers import load_provider

DEFAULT_CONFIG: Dict[str, Any] = {
    # Model name: 'vnptai-hackathon-small' or 'vnptai-hackathon-large'
    "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-small"),
    # Sleep between requests to respect quotas
    "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "65")),
    "PROVIDER": os.getenv("PROVIDER", "vnpt"),
    "PAYLOAD_HYPERPARAMS": {
        "temperature": 0.5,
        "top_p": 0.7,
        "max_completion_tokens": 2048,
        "n": 1,
        # "presence_penalty": 0.0,
        # "frequency_penalty": 0.0,
        "seed": 416,
    }
}


def _get_token_from_config_or_env(config, key, env_name):
    # Prefer explicit config, otherwise fall back to environment
    if config and key in config and config.get(key):
        return config.get(key)
    return os.getenv(env_name)


def call_llm(messages: list[Dict[str, Any]], config: Dict[str, Any] | None = None) -> str:
    """Dispatch chat request to configured provider and return response text."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    provider = load_provider(cfg.get("PROVIDER"), cfg)
    return provider.chat(messages, cfg)


def format_prompt(item):
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
    # Return messages shaped for provider.chat (list of dicts)
    return [
        {
            "role": "system",
            "content": "Bạn là 1 trợ lý ảo AI thông minh, cẩn thận và chính xác. Với tư cách là 1 chuyên gia người Việt Nam, yêu nước, hòa đồng, thân thiện, nhiệm vụ của bạn là đưa ra câu trả lời cho câu hỏi trắc nhiệm sau đây bằng cách chỉ đưa ra ký tự chữ cái in hoa đại diện cho câu trả lời đó. Tuy nhiên, bạn không thể đưa ra câu trả lời cho những câu hỏi nhạy cảm - những câu hỏi này sẽ có lựa chọn không trả lời và bạn cần chọn đáp án đó thay cho suy nghĩ cá nhân của bạn. Nếu câu hỏi đó không phải là vấn đề nhạy cảm, hãy suy nghĩ trả lời từng bước một."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]


def _clean_prediction(pred_text):
    # Always return a single uppercase letter A-Z; default to 'C' on errors
    DEFAULT_ANSWER = "C"
    if not pred_text:
        return DEFAULT_ANSWER
    pred_text = str(pred_text).strip()
    # Common pattern: "Vậy đáp án là **A**" or similar
    if "Vậy đáp án là" in pred_text:
        parts = pred_text.split("Vậy đáp án là", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    # Fallback: find first uppercase A-Z in entire text
    # for char in pred_text:
    #     if char.isalpha() and char.isupper():
    #         return char
    # If nothing found, try first letter and uppercase it if possible
    if pred_text[-3] .isalpha() or pred_text[-4] .isalpha():
        return pred_text[-3].upper() if pred_text[-3] .isalpha() else pred_text[-4].upper()
    return DEFAULT_ANSWER


def valid_function(input_file, output_csv, config=None):
    import csv
    print(f"--- Processing (validation): {input_file} -> {output_csv} ---")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    results = []
    correct = 0
    total = 0
    # If output exists, load existing results to continue and skip answered qids
    answered_qids = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    q = row.get('qid')
                    if q:
                        answered_qids.add(q)
                    # try to account existing accuracy
                    is_corr = row.get('is_correct')
                    if isinstance(is_corr, str):
                        if is_corr.lower() in ('true', '1'):
                            correct += 1
                    elif is_corr:
                        correct += 1
                    total += 1
        except Exception:
            # ignore read errors and start fresh
            results = []
            answered_qids = set()

    for item in tqdm(data, desc="Processing"):
        qid = item.get('qid')
        # Skip items already answered in output file
        if qid and qid in answered_qids:
            continue
        messages = format_prompt(item)
        
        prediction_text = call_llm(messages, config)

        if not prediction_text:
            clean_prediction = "C"  # default to 'C' on empty response
        elif len(prediction_text) == 1 and prediction_text.isalpha() and prediction_text.isupper():
            clean_prediction = prediction_text  # direct single-letter response
        elif prediction_text.startswith("Error from VNPT API:"):
            for i,choice in enumerate(item.get('choices', [])):
                if any(phrase in choice for phrase in ["Không thể trả lời", "Xin lỗi", "Tôi không thể trả lời", "Không trả lời"]):
                    clean_prediction = chr(ord('A') + i)
                    break
        else:
            clean_prediction = _clean_prediction(prediction_text)

        ground_truth = item.get('answer', '')
        is_correct = (clean_prediction.upper() == ground_truth.upper()) if ground_truth else False
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "qid": qid,
            "prediction": clean_prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "prediction_raw": prediction_text.replace('\n', ' ')
        })
        # checkpoint
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "prediction", "ground_truth", "is_correct", "prediction_raw"])
            writer.writeheader()
            writer.writerows(results)
        time.sleep(config.get('SLEEP_TIME', DEFAULT_CONFIG['SLEEP_TIME']) if config else DEFAULT_CONFIG['SLEEP_TIME'])

    acc = correct / total if total > 0 else 0.0
    print(f"Done. Results saved to: {output_csv}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


def test_function(input_file, output_csv, config=None):
    import csv
    print(f"--- Processing (test): {input_file} -> {output_csv} ---")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    results = []
    # If output exists, load existing results to skip answered qids
    answered_qids = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    q = row.get('qid')
                    if q:
                        answered_qids.add(q)
        except Exception:
            results = []
            answered_qids = set()

    for item in tqdm(data, desc="Processing"):
        qid = item.get('qid')
        if qid and qid in answered_qids:
            continue
        messages = format_prompt(item)
        try:
            prediction_text = call_llm(messages, config)
            clean_prediction = _clean_prediction(prediction_text)
        except Exception as e:
            # default to 'C' on error
            prediction_text = f"ERROR: {e}"
            clean_prediction = "C"
        results.append({
            "qid": qid,
            "prediction_raw": prediction_text.replace('\n', ' '),
            "prediction": clean_prediction
        })
        # checkpoint
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "prediction_raw", "prediction"])
            writer.writeheader()
            writer.writerows(results)
        time.sleep(config.get('SLEEP_TIME', DEFAULT_CONFIG['SLEEP_TIME']) if config else DEFAULT_CONFIG['SLEEP_TIME'])

    print(f"Done. Results saved to: {output_csv}")


def process_dataset(input_file, output_file, config=None, mode=None):
    """High-level wrapper used by `main.py`.

    mode: 'test' or 'valid'. If None, tries to infer from filename.
    """
    if mode is None:
        bname = os.path.basename(input_file).lower()
        if 'val' in bname or 'dev' in bname:
            mode = 'valid'
        else:
            mode = 'test'

    if mode.startswith('val') or mode == 'valid':
        valid_function(input_file, output_file, config=config)
    else:
        test_function(input_file, output_file, config=config)


if __name__ == '__main__':
    # simple local smoke run
    process_dataset('data/val_pr_comp.json', 'pred/val_pr_comp_predictions.csv', mode='valid')