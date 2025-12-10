import json
import requests
import time
import os
from tqdm import tqdm

CONFIG = {
    "API_BASE_URL": "https://api.idg.vnpt.vn/data-service/v1/chat/completions",
    "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjA1OWM0Y2U5LTdlOTctNDU3OS1hNjZhLWY5MGZkNzAzOGExMCIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiIzMzViOTVjOC0xOTRmLTRkNTUtOTMxOS0yZThiNjI4OWYyYWEiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.op9nmfwlE7Ekkj8oYLLiSDocrkLFYy45D5SJh5p2rvmdgmGRccJwznPwPsQ-EoRrpaJ0R4NN-v2klEG-0inufgCvkfkTt8wGZeWbjqBEPed6DRf2y1LR7cwf63YrlpFK1ArVcTxrcGkXEoo39mnKBnxlxsrO9IVIaV1QGEnB54jItA62uqz9rBRtf5roJm2IIOXd418SeKl6SPwCYinTgYtswIJ5o2KCahcq-dXuqCrhCmrOGYDhy-eKDZVE-XiR2v4xChaCGU6PUO6F919mN6-otsiJ7DNidB_Ovvl1hVzy5u-mgWJEkv06RZp7rPequFDNzrYXFCwJ5OIM9DQkJg",  # Điền Access Token (Bearer)
    "TOKEN_ID": "4525a88b-e7db-4f0c-e063-62199f0a3a11",          # Điền Token-id
    "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJivUf+ovda9JbCzUkcrs7mHHaNMDmDJK+Hz0yexuxuGjUztbqmfdCIPJGBaGMkRscI4GYtx5p09WCpigc/QkdkCAwEAAQ==",        # Điền Token-key
    
    # Chọn model: 'vnptai-hackathon-small' hoặc 'vnptai-hackathon-large'
    "MODEL_NAME": "vnptai-hackathon-small", 
    
    # Thời gian nghỉ giữa các request (giây) để tránh lỗi Quota
    # Small: 60 req/h -> nghỉ > 60s. Large: 40 req/h -> nghỉ > 90s.
    "SLEEP_TIME": 65 
}

def call_vnpt_llm(prompt, config):
    url = f"{config['API_BASE_URL']}/{config['MODEL_NAME']}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['ACCESS_TOKEN']}",
        "Token-id": config['TOKEN_ID'],
        "Token-key": config['TOKEN_KEY']
    }
    
    model_body_name = config['MODEL_NAME'].replace("-", "_")

    payload = {
        "model": model_body_name,
        "messages": [
            {
                "role": "system",
                "content": "Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là trả lời các câu hỏi trắc nghiệm. Hãy suy nghĩ kỹ và chỉ đưa ra đáp án đúng duy nhất (A, B, C, hoặc D)."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1, 
        "top_p": 1.0,
        "max_completion_tokens": 512,
        "n": 1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Kiểm tra lỗi HTTP (4xx, 5xx)
        
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        else:
            return "Error: No content in response"
            
    except requests.exceptions.RequestException as e:
        return f"HTTP Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def format_prompt(item):
    question = item['question']
    choices = item['choices']
    
    # Gán nhãn A, B, C, D cho các lựa chọn
    labels = ["A", "B", "C", "D", "E", "F"]
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            formatted_choices.append(f"{labels[i]}. {choice}")
    
    prompt_text = f"""
        Câu hỏi: {question}
        Các lựa chọn:
        {chr(10).join(formatted_choices)}

        Yêu cầu: Hãy chọn đáp án đúng nhất cho câu hỏi trên.
        Trả lời (chỉ ghi một chữ cái in hoa A, B, C, hoặc D tương ứng với đáp án đúng):
    """
    return prompt_text

def process_dataset(input_file, output_file, config):
    print(f"--- Đang xử lý file: {input_file} ---")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    processed_ids = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                processed_ids = {item['qid'] for item in results}
                print(f"Tìm thấy file kết quả cũ. Đã xử lý {len(processed_ids)} câu.")
            except json.JSONDecodeError:
                pass

    # 3. Vòng lặp xử lý
    for item in tqdm(data, desc="Processing"):
        qid = item['qid']
        
        if qid in processed_ids:
            continue # Bỏ qua câu đã làm
            
        prompt = format_prompt(item)
        
        # Gọi API
        prediction_text = call_vnpt_llm(prompt, config)
        
        # Xử lý kết quả thô để lấy A, B, C, D (Hậu xử lý đơn giản)
        clean_prediction = prediction_text.strip().split('.')[0].split(' ')[0].strip()
        
        # Lưu kết quả
        result_item = {
            "qid": qid,
            "prediction_raw": prediction_text, # Lưu câu trả lời gốc để debug
            "prediction": clean_prediction
        }
        
        # Nếu là tập val có đáp án, có thể lưu thêm để so sánh sau này
        if "answer" in item:
            result_item["ground_truth"] = item["answer"]

        results.append(result_item)
        
        # Ghi file liên tục (checkpointing) để an toàn dữ liệu
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # NGỦ ĐỂ TRÁNH RATE LIMIT
        time.sleep(config['SLEEP_TIME'])

    print(f"--- Hoàn tất! Kết quả lưu tại: {output_file} ---")


if __name__ == "__main__":
    process_dataset(
        input_file='test.json', 
        output_file='test_predictions.json', 
        config=CONFIG
    )
    
    # Chạy trên tập Val (Bỏ comment dòng dưới nếu muốn chạy cả val)
    process_dataset(
        input_file='val.json', 
        output_file='val_predictions.json', 
        config=CONFIG
    )