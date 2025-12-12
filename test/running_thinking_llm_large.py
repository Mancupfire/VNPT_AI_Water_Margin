import json
import os
import time
import re
import requests
from tqdm import tqdm

CONFIG = {
    # vnptai-hackathon-small
    "API_BASE_URL": "https://api.idg.vnpt.vn/data-service/v1/chat/completions",
    "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjA1OWM0Y2U5LTdlOTctNDU3OS1hNjZhLWY5MGZkNzAzOGExMCIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiIzMzViOTVjOC0xOTRmLTRkNTUtOTMxOS0yZThiNjI4OWYyYWEiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.op9nmfwlE7Ekkj8oYLLiSDocrkLFYy45D5SJh5p2rvmdgmGRccJwznPwPsQ-EoRrpaJ0R4NN-v2klEG-0inufgCvkfkTt8wGZeWbjqBEPed6DRf2y1LR7cwf63YrlpFK1ArVcTxrcGkXEoo39mnKBnxlxsrO9IVIaV1QGEnB54jItA62uqz9rBRtf5roJm2IIOXd418SeKl6SPwCYinTgYtswIJ5o2KCahcq-dXuqCrhCmrOGYDhy-eKDZVE-XiR2v4xChaCGU6PUO6F919mN6-otsiJ7DNidB_Ovvl1hVzy5u-mgWJEkv06RZp7rPequFDNzrYXFCwJ5OIM9DQkJg",  # Điền Access Token (Bearer)
    "TOKEN_ID": "4525a88b-e7db-4f0c-e063-62199f0a3a11",          # Điền Token-id
    "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJivUf+ovda9JbCzUkcrs7mHHaNMDmDJK+Hz0yexuxuGjUztbqmfdCIPJGBaGMkRscI4GYtx5p09WCpigc/QkdkCAwEAAQ==",    
    
    # vnptai-hackathon-large
    # "API_BASE_URL": "https://api.idg.vnpt.vn/data-service/v1/chat/completions",
    # "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjFjOGU0ODFmLTMzYjQtNGFiZC1hMjIxLTM1NTBiZWJhMzA5YiIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiJhNWE1M2UxNi1lNWE2LTQ5YTEtYjhhMy03YmE4ZmZlOTJjMzMiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.uOkouzqVgQij4_hW7kbj3FKkn99mBQR2qtu9EvNQsbqwKlG_Oy_1w2VOkb7QZXyvzwyuXI_2JEA3ekGbKoCy7C6tsa9VkNaYB34GJVMcIPs6YCjkOJK6ktH4HhMEYmH7xdY0jOYj8OTFdv7CxhxTaFmFQzLPWEGrVcD5dppM0Ci4GDZ6WfRWv87QC4KGZw8j5M5GtIL9_3PvDNbRNjpHb9l5JF_a-tBp772YgcjuE9heGNlcQ5EPdOlUBWQdJdZwIEPcoS3IPUYi5cvSOX08Gu04LALSsIArxRcLn0s44K1YyUuN67NcH0gimVQSOaz-zPvwpFqk9xK0vrPTjpJPlw",  # Điền Access Token (Bearer)
    # "TOKEN_ID": "4525a84b-002f-2031-e063-62199f0af9db",          # Điền Token-id
    # "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIKOkkLUohQIfhfL42rRUyJqc9GVrj42P6/Z9EHl/NnRM19yI7TnVrhXK9pzBhBNS4L6Ks6ohcrlIwqf2CE6rr0CAwEAAQ==",        
    
    # Chọn model: 'vnptai-hackathon-small' hoặc 'vnptai-hackathon-large'
    "MODEL_NAME": "vnptai-hackathon-small",
    
    # Thời gian nghỉ giữa các request (giây) để tránh lỗi Quota
    # Small: 60 req/h -> nghỉ > 60s. Large: 40 req/h -> nghỉ > 90s.
    "SLEEP_TIME": 20
}

# ==============================================================================
# 1. HÀM GỌI API
# ==============================================================================

def call_vnpt_llm(system_prompt, user_prompt, config):
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
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.1, 
        "top_p": 1.0,
        "max_completion_tokens": 2048,
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

# ==============================================================================
# 2. CÁC HÀM TẠO PROMPT (PROMPT TEMPLATES)
# ==============================================================================

def format_choices(choices):
    formatted = []
    for i, choice in enumerate(choices):
        label = chr(65 + i) # A, B, C...
        formatted.append(f"{label}. {choice}")
    return "\n".join(formatted)

def prompt_safety_refusal(question, choices_text):
    system = "Bạn là một trợ lý AI tuân thủ nghiêm ngặt các tiêu chuẩn an toàn (Safety & Ethics)."
    user = f"""
    # Task
    Câu hỏi dưới đây có thể vi phạm tiêu chuẩn an toàn. Nhiệm vụ của bạn là TỪ CHỐI trả lời nội dung đó bằng cách **chọn đáp án thể hiện sự từ chối** trong danh sách.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Giải thích ngắn gọn lý do từ chối.",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C...)"
    }}
    """
    return system, user

def prompt_reading_comprehension(question, choices_text):
    system = "Bạn là một chuyên gia đọc hiểu văn bản. Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm dựa **TUYỆT ĐỐI** và **DUY NHẤT** vào đoạn văn bản được cung cấp."
    user = f"""
    # Rules
    1. KHÔNG sử dụng kiến thức bên ngoài.
    2. Tìm bằng chứng (Evidence) trong văn bản.
    3. **QUAN TRỌNG: Phần suy luận hãy viết ngắn gọn, súc tích. KHÔNG chép lại nguyên văn các đoạn văn dài.**

    # Input
    {question}

    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "step_by_step_reasoning": "Tóm tắt ngắn gọn manh mối tìm được -> Suy luận logic -> Kết luận.",
    "final_answer": "Ký tự in hoa của đáp án đúng"
    }}
    """
    return system, user

def prompt_math_logic(question, choices_text):
    system = "Bạn là một Giáo sư thông thái, chuyên gia giải quyết các bài toán định lượng thuộc các lĩnh vực: Toán học, Vật lý, Hóa học và Tài chính/Kế toán. Tư duy của bạn mạch lạc, chặt chẽ và luôn tuân thủ logic từng bước (Chain of Thought)."
    user = f"""
    # Nhiệm vụ
    Giải quyết bài toán trắc nghiệm dưới đây và trả về kết quả dưới dạng JSON.

    # Hướng dẫn xử lý (Heuristics quan trọng):
    1. **Tài chính/Lãi suất:** - Mặc định thử tính **Lãi kép (Compound)** trước.
    - Nếu kết quả không khớp, HÃY TÍNH LẠI bằng **Lãi đơn (Simple Interest)** (đặc biệt là các bài toán nợ ngắn hạn hoặc kế toán cơ bản).
    2. **Vật lý/Hóa học:**
    - Chú ý đổi đơn vị (ví dụ: cm -> m, gram -> kg, phút -> giây) trước khi tính.
    - Nếu kết quả lệch nhẹ (do lấy g=9.8 hay g=10), hãy chọn đáp án có giá trị gần nhất.
    3. **So sánh đáp án:**
    - Bỏ qua các ký tự định dạng (như $, %, _, \\frac) trong các lựa chọn. Chỉ so sánh giá trị số.
    - Ví dụ: Tính ra 0.75m0 thì chọn đáp án "$0.75m_0c$".

    # Ví dụ mẫu (One-shot Learning):
    Input:
    Câu hỏi: Một vật rơi tự do trong 2s, lấy g=10m/s^2. Quãng đường là?
    A. 10m
    B. 20m
    Output JSON:
    {{
    "step_by_step_reasoning": "Công thức s = 0.5 * g * t^2. Thay số: 0.5 * 10 * 2^2 = 0.5 * 10 * 4 = 20. Kết quả là 20m. Khớp chính xác với đáp án B.",
    "final_answer": "B"
    }}

    # Input Dữ liệu thực tế:
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON Only):
    {{
    "step_by_step_reasoning": "Phân tích đề -> Chọn công thức -> Thử phương pháp 1 (Lãi kép/g=9.8) -> Nếu lệch thì thử phương pháp 2 (Lãi đơn/g=10) -> Chốt kết quả.",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C...)"
    }}
    """
    return system, user

def prompt_vn_core_knowledge(question, choices_text):
    system = "Bạn là chuyên gia về Chính trị, Pháp luật và Lịch sử Việt Nam. Bạn trả lời dựa trên quan điểm chính thống và văn bản pháp luật hiện hành."
    user = f"""
    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Giải thích dựa trên kiến thức chính thống...",
    "final_answer": "Ký tự in hoa của đáp án đúng"
    }}
    """
    return system, user

def prompt_general_domain(question, choices_text):
    system = "Bạn là một trợ lý AI thông thái với kiến thức bách khoa."
    user = f"""
    # Task
    Chọn đáp án chính xác nhất bằng tư duy logic và kiến thức phổ quát.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Phân tích logic...",
    "final_answer": "Ký tự in hoa của đáp án đúng"
    }}
    """
    return system, user

# Hàm điều phối (Dispatcher) trả về (system, user)
def get_prompt_by_domain(item):
    domain = item.get('domain_label', 'GENERAL_DOMAIN') 
    question = item['question']
    choices_text = format_choices(item['choices'])
    
    if domain == 'SAFETY_REFUSAL':
        return prompt_safety_refusal(question, choices_text)
    elif domain == 'READING_COMPREHENSION':
        return prompt_reading_comprehension(question, choices_text)
    elif domain == 'MATH_LOGIC':
        return prompt_math_logic(question, choices_text)
    elif domain == 'VN_CORE_KNOWLEDGE':
        return prompt_vn_core_knowledge(question, choices_text)
    else: 
        return prompt_general_domain(question, choices_text)

# ==============================================================================
# 3. HÀM XỬ LÝ CHÍNH (MAIN PROCESS)
# ==============================================================================

def extract_json_from_response(text):
    """
    Trích xuất JSON bằng cách đếm cân bằng ngoặc nhọn {}, 
    giúp tránh lỗi khi văn bản chứa ký tự } của LaTeX.
    """
    text = text.strip()
    
    # 1. Tìm vị trí bắt đầu của JSON
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # 2. Thuật toán Stack để tìm dấu đóng } tương ứng
    balance = 0
    is_inside_string = False
    escape = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        # Xử lý dấu ngoặc kép để tránh đếm nhầm {} bên trong chuỗi string
        if char == '"' and not escape:
            is_inside_string = not is_inside_string
        
        if char == '\\' and not escape:
            escape = True
        else:
            escape = False
            
        if not is_inside_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                
            # Khi balance về 0, tức là đã đóng đủ ngoặc cho JSON object đầu tiên
            if balance == 0:
                json_str = text[start_idx : i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Nếu parse lỗi, có thể do cấu trúc bên trong sai, tiếp tục thử (hiếm gặp)
                    continue
                    
    return None

def process_answering_dataset(original_file, labeled_file, output_file, config, limit=None):
    print(f"--- Bắt đầu trả lời câu hỏi ---")
    
    # 1. Load dữ liệu
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {original_file}")
        return

    domain_map = {}
    try:
        with open(labeled_file, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
            for item in labeled_data:
                domain_map[item['qid']] = item.get('predicted_domain', 'GENERAL_DOMAIN')
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy file nhãn. Dùng mặc định GENERAL_DOMAIN.")

    # 2. Cắt limit nếu test
    if limit is not None:
        print(f"CHẾ ĐỘ TEST: Chỉ xử lý {limit} câu đầu tiên.")
        original_data = original_data[:limit]

    # 3. Checkpoint
    processed_ids = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                processed_ids = {item['qid'] for item in results}
                print(f"Đã xử lý trước đó: {len(processed_ids)} câu.")
            except json.JSONDecodeError:
                pass

    # 4. Vòng lặp xử lý
    for item in tqdm(original_data, desc="Answering"):
        qid = item['qid']
        if qid in processed_ids:
            continue

        # Inject domain
        item['domain_label'] = domain_map.get(qid, 'GENERAL_DOMAIN')

        # Lấy System và User prompt
        system_msg, user_msg = get_prompt_by_domain(item)

        # --- GỌI HÀM CỦA BRO Ở ĐÂY ---
        response_text = call_vnpt_llm(system_msg, user_msg, config)
        # -----------------------------

        # Hậu xử lý kết quả
        parsed_json = extract_json_from_response(response_text)
        
        final_ans = ""
        reasoning = ""
        
        if parsed_json:
            final_ans = parsed_json.get("final_answer", "")
            reasoning = parsed_json.get("reasoning", "") or parsed_json.get("step_by_step_reasoning", "")
        else:
            # Fallback nếu model không trả JSON
            match = re.search(r'\b([A-Z])\b', response_text[-50:])
            if match:
                final_ans = match.group(1)
            reasoning = response_text 

        # Lưu kết quả
        result_item = {
            "qid": qid,
            "domain": item['domain_label'],
            "llm_prediction": final_ans,
            "llm_reasoning": reasoning,
            "ground_truth": item.get("answer", "")
        }
        
        results.append(result_item)

        # Ghi file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        time.sleep(config.get('SLEEP_TIME', 1))

    print(f"--- Hoàn tất! Kết quả lưu tại: {output_file} ---")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    process_answering_dataset(
        original_file='data/val.json',                
        labeled_file='results/val_classification.json',
        output_file='results/val_answers_small.json',
        config=CONFIG
    )
    
    # process_answering_dataset(
    #     original_file='data/test.json', 
    #     labeled_file='results/test_classification.json', 
    #     output_file='results/test_answers.json', 
    #     config=CONFIG
    # )

    # process_answering_dataset(
    #     original_file='data/draft.json', 
    #     labeled_file='results/draft_classification.json', 
    #     output_file='results/draft_answers.json', 
    #     config=CONFIG,
    # )