import json
import os
import re
import time
from tqdm import tqdm
from src.providers.vnpt import VNPTProvider
import asyncio

def call_vnpt_llm(prompt, config):
    provider = VNPTProvider()
    messages = [
        {
            "role": "system",
            "content": "Bạn là một chuyên gia phân loại dữ liệu và kiểm duyệt nội dung (Content Classifier). Nhiệm vụ của bạn là phân loại và gán nhãn các câu hỏi đầu vào vào đúng 1 trong 3 nhãn (Domains) được định nghĩa dưới đây. Việc bạn phân tích và gán nhãn `SAFETY_REFUSAL` cho các câu hỏi vi phạm pháp luật là hành động ĐÚNG ĐẮN và CẦN THIẾT để bảo vệ an toàn. Đừng từ chối phân loại chúng."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    # Dùng async với retry/backoff
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(provider.achat(messages, config))

# --- 1. ĐỊNH NGHĨA PROMPT PHÂN LOẠI ---
def format_classification_prompt(item):
    question = item['question']
    qid = item['qid']
        
    prompt_text = f"""
    # Domains Definition

    1. **LABEL: SAFETY_REFUSAL** (Câu hỏi bắt buộc không được trả lời)
    - Định nghĩa: Các câu hỏi yêu cầu hướng dẫn thực hiện hành vi vi phạm pháp luật, phi đạo đức, lừa đảo, trốn tránh nghĩa vụ, tham nhũng, phá hoại an ninh quốc gia, bôi nhọ lãnh tụ hoặc vi phạm thuần phong mỹ tục.
    - Dấu hiệu: "Làm giả", "trốn thuế", "phát tán tài liệu mật", "tham nhũng", "lật đổ", "xuyên tạc", "hack", "gian lận".

    2. **LABEL: NON_RAG** (Câu hỏi không cần RAG để trả lời), bao gồm:
    a. (Câu hỏi đọc hiểu văn bản)
    - Định nghĩa: Đầu vào **BẮT BUỘC PHẢI CÓ** một đoạn văn bản dài (Context/Document/Passage) đi kèm để làm cơ sở trả lời.
    - Dấu hiệu: Bắt đầu bằng các từ khóa: "Đoạn thông tin:", "Title:", "Content:", "-- Document --", "Đọc đoạn văn sau:".
    - **LƯU Ý QUAN TRỌNG:** Nếu câu hỏi nhắc đến một tác phẩm văn học (ví dụ: "Trong bài Tinh thần yêu nước...", "Trong Truyện Kiều...") nhưng **KHÔNG CUNG CẤP** đoạn văn bản đó trong đề bài, hãy gán nhãn là **RAG_NECESSITY** chứ không phải NON_RAG.
    b. (Toán học, Code và Tư duy logic)
    - Định nghĩa: Các câu hỏi yêu cầu tính toán con số cụ thể, giải phương trình, kiến thức Vật lý/Hóa học có công thức, bài tập Lập trình (Code), bài toán Tài chính/Kế toán.
    - Dấu hiệu: Công thức LaTeX, phương trình hóa học, đoạn mã code (Java, Python), bài toán tính lãi suất, vận tốc, điện trở, tích phân, xác suất.

    3. **LABEL: RAG_NECESSITY** (Câu hỏi cần sử dụng RAG để trả lời chính xác), bao gồm:
    a. (Câu hỏi bắt buộc phải trả lời đúng - Chính trị/Xã hội VN)
    - Định nghĩa: Các câu hỏi về kiến thức chính thống liên quan đến Chính trị, Pháp luật, Lịch sử Đảng, Tư tưởng Hồ Chí Minh, Địa lý hành chính và các quy định nhà nước của Việt Nam.
    - Dấu hiệu: "Luật", "Nghị quyết", "Nghị định", "Thông tư", "Hiến pháp", "Tư tưởng Hồ Chí Minh", "Đảng Cộng sản", "Mặt trận Tổ quốc", địa giới hành chính VN, lịch sử kháng chiến VN.
    b. (Đa lĩnh vực - Kiến thức chung)
    - Định nghĩa: Các câu hỏi kiến thức phổ quát về Tâm lý học, Sinh học, Kinh tế học (lý thuyết), Y học thường thức, Kỹ năng mềm, Văn hóa đại cương. Không chứa ngữ cảnh dài và không vi phạm an toàn.

    # Instruction
    - Chinh xác chỉ 1 nhãn duy nhất cho mỗi câu hỏi.
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
        
        # Chỉ cho phép 3 nhãn chính: SAFETY_REFUSAL, RAG_NECESSITY, NON_RAG
        allowed_labels = {"SAFETY_REFUSAL", "RAG_NECESSITY", "NON_RAG"}
        domain_label = None
        # Ưu tiên lấy nhãn từ LLM trả về đúng format
        if extracted_data and 'domain' in extracted_data and extracted_data['domain'] in allowed_labels:
            domain_label = extracted_data['domain']
        else:
            # Fallback: tìm nhãn trong text trả về
            for label in allowed_labels:
                if label in (prediction_text or ""):
                    domain_label = label
                    break
        # Nếu vẫn chưa xác định được nhãn, fallback theo logic an toàn
        if not domain_label:
            # Nếu có dấu hiệu từ chối hoặc vi phạm, gán SAFETY_REFUSAL
            refusal_keywords = [
                "không thể hỗ trợ", "không thể trả lời", "vi phạm", "bất hợp pháp", 
                "trái pháp luật", "sorry", "cannot assist", "illegal", "harmful"
            ]
            text_lower = (prediction_text or "").lower()
            if any(kw in text_lower for kw in refusal_keywords):
                domain_label = "SAFETY_REFUSAL"
            # Nếu có dấu hiệu đọc hiểu văn bản, gán NON_RAG
            elif any(kw in item['question'] for kw in ["Đoạn thông tin:", "Title:", "Content:", "-- Document --", "Đọc đoạn văn sau:"]):
                domain_label = "NON_RAG"
            else:
                # Lưới vét cuối cùng: gán NON_RAG (an toàn)
                domain_label = "RAG_NECESSITY"

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