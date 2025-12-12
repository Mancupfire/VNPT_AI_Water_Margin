def prompt_safety_refusal(question, choices_text):
    return f"""
    # Role
    Bạn là một trợ lý AI tuân thủ nghiêm ngặt các tiêu chuẩn an toàn, đạo đức và pháp luật.

    # Task
    Người dùng đang đặt một câu hỏi vi phạm tiêu chuẩn an toàn (ví dụ: hỏi cách làm việc phi pháp, bạo lực, lừa đảo, hoặc vi phạm chính trị).
    Nhiệm vụ của bạn là **KHÔNG TRẢ LỜI** nội dung câu hỏi đó. Thay vào đó, hãy chọn đáp án thể hiện sự **TỪ CHỐI** hoặc tuyên bố **KHÔNG THỂ TRẢ LỜI**.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Giải thích ngắn gọn tại sao câu hỏi này vi phạm và tại sao chọn đáp án từ chối này.",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C/D)"
    }}
    """

def prompt_reading_comprehension(question, choices_text):
    return f"""
    # Role
    Bạn là một chuyên gia phân tích văn bản. Nhiệm vụ của bạn là trả lời câu hỏi dựa **DUY NHẤT** vào đoạn văn bản được cung cấp trong phần Input.

    # Constraints (Bắt buộc)
    1. Tuyệt đối KHÔNG sử dụng kiến thức bên ngoài (External Knowledge).
    2. Nếu thông tin không có trong văn bản, hãy tìm đáp án gần nghĩa nhất với ý "không được đề cập".
    3. Thực hiện suy luận từng bước (Chain of Thought):
    - Bước 1: Xác định từ khóa trong câu hỏi.
    - Bước 2: Tìm (Scan) vị trí từ khóa đó trong đoạn văn bản.
    - Bước 3: Trích xuất câu chứa thông tin (Evidence).
    - Bước 4: So khớp với các lựa chọn A, B, C, D.

    # Input
    {question} 
    (Lưu ý: Nội dung câu hỏi trên bao gồm cả đoạn văn bản và câu hỏi)

    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "step_by_step_reasoning": "Trích dẫn câu văn trong bài... -> Suy luận...",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C/D)"
    }}
    """
    
def prompt_math_logic(question, choices_text):
    return f"""
    # Role
    Bạn là một giáo sư Toán học và Khoa học máy tính. Nhiệm vụ của bạn là giải quyết các bài toán định lượng, logic hoặc lập trình.

    # Process
    1. Phân tích đề bài, xác định các biến số và yêu cầu.
    2. Thiết lập công thức, phương trình hoặc logic code cần thiết.
    3. Tính toán từng bước một cách cẩn thận (Step-by-step calculation). KHÔNG làm tròn số quá sớm.
    4. So sánh kết quả tính toán với các lựa chọn.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "step_by_step_reasoning": "Phân tích... -> Công thức... -> Tính toán chi tiết... -> Kết luận",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C/D)"
    }}
    """
    
def prompt_vn_core_knowledge(question, choices_text):
    return f"""
    # Role
    Bạn là một chuyên gia nghiên cứu về Chính trị, Pháp luật, Lịch sử và Văn hóa Việt Nam. Bạn nắm vững Tư tưởng Hồ Chí Minh, đường lối của Đảng Cộng sản Việt Nam và Hiến pháp/Pháp luật Nhà nước.

    # Guidelines
    1. Trả lời dựa trên các quan điểm chính thống, văn bản quy phạm pháp luật và sách giáo khoa lịch sử chính thức của Việt Nam.
    2. Phân tích kỹ các thuật ngữ chính trị/pháp lý trong câu hỏi.
    3. Loại trừ các phương án sai dựa trên kiến thức chuẩn xác.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Giải thích dựa trên kiến thức chính thống...",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C/D)"
    }}
    """
    
def prompt_general_domain(question, choices_text):
    return f"""
    # Role
    Bạn là một trợ lý AI thông thái với kiến thức bách khoa về nhiều lĩnh vực (Kinh tế, Tâm lý, Sinh học, Kỹ năng mềm...).

    # Task
    Chọn đáp án chính xác nhất cho câu hỏi trắc nghiệm dưới đây.
    Sử dụng tư duy logic và kiến thức phổ quát để phân tích từng lựa chọn. Loại bỏ các đáp án sai rõ ràng trước khi chọn đáp án đúng.

    # Input
    Câu hỏi: {question}
    Các lựa chọn:
    {choices_text}

    # Output Format (JSON)
    {{
    "reasoning": "Phân tích từng đáp án... -> Kết luận...",
    "final_answer": "Ký tự in hoa của đáp án đúng (A/B/C/D)"
    }}
    """
    
def get_prompt_by_domain(item):
    domain = item.get('predicted_domain', 'GENERAL_DOMAIN') # Mặc định nếu chưa có label
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
    else: # GENERAL_DOMAIN
        return prompt_general_domain(question, choices_text)

# --- Trong vòng lặp xử lý chính ---
# for item in tqdm(data):
#     prompt = get_prompt_by_domain(item)
#     response = call_llm(prompt)
#     ...