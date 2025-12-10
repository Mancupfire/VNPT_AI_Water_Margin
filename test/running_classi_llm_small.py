import json
import requests
import time
import os
import re
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
                "content": "Bạn là một chuyên gia phân loại dữ liệu và kiểm duyệt nội dung (Content Classifier). Nhiệm vụ của bạn là phân loại các câu hỏi đầu vào vào đúng 1 trong 5 nhãn (Domains) được định nghĩa dưới đây."
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

# --- 1. ĐỊNH NGHĨA PROMPT PHÂN LOẠI ---
def format_classification_prompt(item):
    question = item['question']
    qid = item['qid']
        
    prompt_text = f"""
    # Domains Definition

    1. **LABEL: SAFETY_REFUSAL** (Câu hỏi bắt buộc không được trả lời)
    - Định nghĩa: Các câu hỏi yêu cầu hướng dẫn thực hiện hành vi vi phạm pháp luật, phi đạo đức, lừa đảo, trốn tránh nghĩa vụ, tham nhũng, phá hoại an ninh quốc gia, bôi nhọ lãnh tụ hoặc vi phạm thuần phong mỹ tục.
    - Dấu hiệu: "Làm giả", "trốn thuế", "phát tán tài liệu mật", "tham nhũng", "lật đổ", "xuyên tạc", "hack", "gian lận".

    2. **LABEL: VN_CORE_KNOWLEDGE** (Câu hỏi bắt buộc phải trả lời đúng - Chính trị/Xã hội VN)
    - Định nghĩa: Các câu hỏi về kiến thức chính thống liên quan đến Chính trị, Pháp luật, Lịch sử Đảng, Tư tưởng Hồ Chí Minh, Địa lý hành chính và các quy định nhà nước của Việt Nam.
    - Dấu hiệu: "Luật", "Nghị định", "Tư tưởng Hồ Chí Minh", "Đảng Cộng sản", "Hiến pháp", "Mặt trận Tổ quốc", địa giới hành chính VN, lịch sử kháng chiến VN.

    3. **LABEL: READING_COMPREHENSION** (Câu hỏi đọc hiểu văn bản)
    - Định nghĩa: Đầu vào cung cấp một đoạn văn bản (Context/Document/Passage) dài và yêu cầu trả lời dựa trên thông tin đó.
    - Dấu hiệu: Bắt đầu bằng "Đoạn thông tin:", "Title:", "Content:", "-- Document --", hoặc chứa một đoạn văn dài trước khi đưa ra câu hỏi.

    4. **LABEL: MATH_LOGIC** (Toán học, Code và Tư duy logic)
    - Định nghĩa: Các câu hỏi yêu cầu tính toán con số cụ thể, giải phương trình, kiến thức Vật lý/Hóa học có công thức, bài tập Lập trình (Code), bài toán Tài chính/Kế toán.
    - Dấu hiệu: Công thức LaTeX, phương trình hóa học, đoạn mã code (Java, Python), bài toán tính lãi suất, vận tốc, điện trở, tích phân, xác suất.

    5. **LABEL: GENERAL_DOMAIN** (Đa lĩnh vực - Kiến thức chung)
    - Định nghĩa: Các câu hỏi kiến thức phổ quát về Tâm lý học, Sinh học, Kinh tế học (lý thuyết), Y học thường thức, Kỹ năng mềm, Văn hóa đại cương. Không chứa ngữ cảnh dài và không vi phạm an toàn.

    # Instruction
    - Đọc kỹ `question` đầu vào.
    - Phân tích nội dung và mục đích của câu hỏi.
    - Chỉ trả về kết quả dưới dạng JSON với format: {{"qid": "{qid}", "domain": "TÊN_LABEL"}}
    - Không giải thích gì thêm.

    # Task
    Hãy phân loại câu hỏi sau:
    {question}
    """
    
    return prompt_text

# --- 2. HÀM HẬU XỬ LÝ (POST-PROCESSING) ---
def extract_json_from_response(response_text):
    """
    Cố gắng trích xuất JSON từ phản hồi của LLM (kể cả khi nó bị bao quanh bởi text khác)
    """
    try:
        # Trường hợp 1: Trả về JSON thuần
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Trường hợp 2: Trả về trong block code ```json ... ``` hoặc lẫn trong text
            # Tìm chuỗi bắt đầu bằng { và kết thúc bằng }
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return None
        except:
            return None

# --- 3. HÀM XỬ LÝ CHÍNH ---
def process_classification_dataset(input_file, output_file, config, limit=None):
    print(f"--- Đang phân loại file: {input_file} ---")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    # --- CẮT DỮ LIỆU NẾU CÓ LIMIT ---
    if limit is not None:
        print(f"CHẾ ĐỘ TEST: Chỉ xử lý {limit} câu đầu tiên.")
        data = data[:limit] # Cắt list data, chỉ lấy từ 0 đến limit
    # -------------------------------

    # Load checkpoint cũ nếu có
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

    # Vòng lặp xử lý
    for item in tqdm(data, desc="Classifying"):
        qid = item['qid']
        
        # Nếu đang test (có limit) mà câu này đã làm rồi thì vẫn nên in ra hoặc bỏ qua tùy logic,
        # nhưng thường test thì mình muốn chạy lại hoặc chạy mới.
        # Ở đây giữ nguyên logic check ID để tránh trùng lặp.
        if qid in processed_ids:
            continue 
            
        prompt = format_classification_prompt(item)
        
        # Gọi API
        prediction_text = call_vnpt_llm(prompt, config)
        
        # Xử lý kết quả
        extracted_data = extract_json_from_response(prediction_text)
        
        if extracted_data and 'domain' in extracted_data:
            domain_label = extracted_data['domain']
        else:
            domain_label = "UNKNOWN" 
            # Fallback logic đơn giản
            if "SAFETY_REFUSAL" in prediction_text: domain_label = "SAFETY_REFUSAL"
            elif "VN_CORE_KNOWLEDGE" in prediction_text: domain_label = "VN_CORE_KNOWLEDGE"
            elif "READING_COMPREHENSION" in prediction_text: domain_label = "READING_COMPREHENSION"
            elif "MATH_LOGIC" in prediction_text: domain_label = "MATH_LOGIC"
            elif "GENERAL_DOMAIN" in prediction_text: domain_label = "GENERAL_DOMAIN"

        # Lưu kết quả
        result_item = {
            "qid": qid,
            "question_snippet": item['question'][:50] + "...",
            "predicted_domain": domain_label,
            # "llm_raw_response": prediction_text # Có thể bỏ comment dòng này để debug kỹ hơn khi test
        }
        
        # Nếu file gốc có đáp án (như file val), lưu lại để tiện so sánh
        if "answer" in item:
             result_item["ground_truth_option"] = item["answer"]

        results.append(result_item)
        
        # Ghi file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        time.sleep(config.get('SLEEP_TIME', 1))

    print(f"--- Hoàn tất! Kết quả (test {limit if limit else 'all'}) lưu tại: {output_file} ---")

# --- VÍ DỤ CÁCH GỌI HÀM ---
# process_classification_dataset('test.json', 'test_labeled.json', config)
# process_classification_dataset('val.json', 'val_labeled.json', config)


if __name__ == "__main__":
    # process_classification_dataset(
    #     input_file='data/test.json', 
    #     output_file='results/test_classification.json', 
    #     config=CONFIG,
    #     limit=5
    # )
    
    # Chạy trên tập Val (Bỏ comment dòng dưới nếu muốn chạy cả val)
    process_classification_dataset(
        input_file='data/val.json', 
        output_file='results/val_classification.json', 
        config=CONFIG,
    )
    
    # process_classification_dataset(
    #     input_file='data/draft.json', 
    #     output_file='results/draft_classification.json', 
    #     config=CONFIG,
    #     limit=1
    # )