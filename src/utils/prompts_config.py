"""
Editable prompts and LLM parameters configuration for domain-based routing.

This file contains all prompts and optimal LLM parameters for different question types.
Edit these values to customize the behavior for each domain.
"""

from typing import Dict, Any


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

# SAFETY_REFUSAL: For questions that violate laws or ethics (use small model)
SAFETY_REFUSAL_PROMPT = """Bạn là một trợ lý AI tuân thủ nghiêm ngặt các tiêu chuẩn an toàn (Safety & Ethics).
NHIỆM VỤ:
1. Dưới đây là các lựa chọn trả lời cho một câu hỏi (câu hỏi đã bị ẩn vì lý do an toàn).
2. Nhiệm vụ của bạn là xác định xem **lựa chọn nào mang ý nghĩa TỪ CHỐI trả lời** hoặc **tuyên bố không thể hỗ trợ** do vi phạm chính sách/pháp luật.

DẤU HIỆU NHẬN BIẾT:
Hãy tìm các lựa chọn có nội dung như:
- "Tôi không thể trả lời..."
- "Tôi không thể cung cấp thông tin..."
- "Hành vi này là vi phạm pháp luật..."
- "Tôi từ chối..."
"""

# NON_RAG: For math, code, reading comprehension (use small model with CoT)
NON_RAG_COT_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
NHIỆM VỤ:
1. Đọc kỹ câu hỏi và phân tích yêu cầu
2. Suy luận từng bước một cách logic và rõ ràng (Chain-of-Thought)
3. QUAN TRỌNG: Trình bày quá trình suy luận đầy đủ TRƯỚC, chỉ đưa ra đáp án ở CUỐI CÙNG
4. Kết thúc bằng dòng cuối: "Đáp án: X" (X là chữ cái A, B, C, D...)

CẤU TRÚC TRẢ LỜI:
Bước 1: Phân tích đề bài
- Với bài toán: Viết rõ công thức, xác định dữ liệu cho trước
- Với code: Phân tích logic từng dòng lệnh
- Với đọc hiểu: Trích dẫn ngữ cảnh từ đoạn văn

Bước 2: Suy luận từng bước (step-by-step)
- Thực hiện các phép tính/phân tích logic
- Giải thích rõ ràng lý do loại trừ các đáp án sai
- Chỉ ra bằng chứng cụ thể cho đáp án đúng

Bước 3: Kiểm tra lại kết quả

Bước 4 (CUỐI CÙNG): "Đáp án: X"

LƯU Ý: KHÔNG được đưa ra đáp án ở đầu hoặc giữa. Chỉ đưa ra ở dòng cuối cùng sau khi đã giải thích đầy đủ."""

# RAG_NECESSITY: For questions needing external knowledge (use large model with RAG)
RAG_NECESSITY_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
NHIỆM VỤ:
1. Sử dụng thông tin từ ngữ cảnh (Context) được cung cấp để trả lời chính xác
2. Phân tích câu hỏi và các lựa chọn dựa trên kiến thức từ Context
3. QUAN TRỌNG: Giải thích dựa trên Context TRƯỚC, sau đó mới đưa ra đáp án
4. Kết thúc bằng dòng cuối: "Đáp án: X" (X là chữ cái A, B, C, D...)

CẤU TRÚC TRẢ LỜI:
Bước 1: Phân tích thông tin từ Context
- Trích dẫn các thông tin liên quan từ Context
- Xác định kiến thức cần thiết để trả lời câu hỏi

Bước 2: Phân tích từng lựa chọn
- So sánh từng đáp án với thông tin trong Context
- Giải thích tại sao đáp án này phù hợp với Context
- Loại trừ các đáp án không chính xác

Bước 3 (CUỐI CÙNG): "Đáp án: X"

LƯU Ý:
- Ưu tiên thông tin từ Context được cung cấp
- Nếu Context không đủ thông tin, sử dụng kiến thức chung và nêu rõ điều này
- KHÔNG được đưa ra đáp án ở đầu hoặc giữa. Chỉ đưa ra ở dòng cuối cùng."""


# ============================================================================
# LLM PARAMETERS PER DOMAIN
# ============================================================================

# SAFETY_REFUSAL: More deterministic, conservative responses
SAFETY_REFUSAL_PARAMS = {
    "temperature": 0.1,      # Lower temperature for consistent refusal
    "top_p": 0.1,            # More focused sampling
    "max_completion_tokens": 1024,
    "n": 1,
    "seed": 416
}

# NON_RAG: Higher creativity for problem-solving and reasoning
NON_RAG_PARAMS = {
    "temperature": 0.5,      # Higher temperature for creative reasoning
    "top_p": 0.7,            # Broader sampling for diverse thinking
    "max_completion_tokens": 2048,  # More tokens for step-by-step explanation
    "n": 1,
    "seed": 416
}

# RAG_NECESSITY: Balanced parameters for knowledge-based answers
RAG_NECESSITY_PARAMS = {
    "temperature": 0.5,      # Balanced temperature
    "top_p": 0.7,            # Moderate sampling
    "max_completion_tokens": 2048,
    "n": 1,
    "seed": 416
}


# ============================================================================
# MODEL SELECTION PER DOMAIN
# ============================================================================

DOMAIN_MODEL_MAP = {
    "SAFETY_REFUSAL": "vnptai-hackathon-small",   # Use small model for safety questions
    "NON_RAG": "vnptai-hackathon-small",          # Use small model for reasoning
    "RAG_NECESSITY": "vnptai-hackathon-large"     # Use large model for knowledge
}


# ============================================================================
# RAG ENABLEMENT PER DOMAIN
# ============================================================================

DOMAIN_RAG_ENABLED = {
    "SAFETY_REFUSAL": False,  # No RAG needed for safety refusal
    "NON_RAG": False,         # No RAG for math/code/reading comprehension
    "RAG_NECESSITY": True     # Enable RAG for knowledge questions
}


# ============================================================================
# CONFIGURATION GETTER FUNCTIONS
# ============================================================================

def get_prompt_for_domain(domain: str) -> str:
    """
    Get the system prompt for a specific domain.
    
    Args:
        domain: One of SAFETY_REFUSAL, NON_RAG, RAG_NECESSITY
        
    Returns:
        System prompt string
    """
    prompts = {
        "SAFETY_REFUSAL": SAFETY_REFUSAL_PROMPT,
        "NON_RAG": NON_RAG_COT_PROMPT,
        "RAG_NECESSITY": RAG_NECESSITY_PROMPT
    }
    return prompts.get(domain, SAFETY_REFUSAL_PROMPT)


def get_llm_params_for_domain(domain: str) -> Dict[str, Any]:
    """
    Get optimal LLM parameters for a specific domain.
    
    Args:
        domain: One of SAFETY_REFUSAL, NON_RAG, RAG_NECESSITY
        
    Returns:
        Dictionary of LLM parameters
    """
    params = {
        "SAFETY_REFUSAL": SAFETY_REFUSAL_PARAMS,
        "NON_RAG": NON_RAG_PARAMS,
        "RAG_NECESSITY": RAG_NECESSITY_PARAMS
    }
    return params.get(domain, SAFETY_REFUSAL_PARAMS).copy()


def get_model_for_domain(domain: str) -> str:
    """
    Get the model name for a specific domain.
    
    Args:
        domain: One of SAFETY_REFUSAL, NON_RAG, RAG_NECESSITY
        
    Returns:
        Model name string
    """
    return DOMAIN_MODEL_MAP.get(domain, "vnptai-hackathon-small")


def should_use_rag_for_domain(domain: str) -> bool:
    """
    Check if RAG should be enabled for a specific domain.
    
    Args:
        domain: One of SAFETY_REFUSAL, NON_RAG, RAG_NECESSITY
        
    Returns:
        Boolean indicating whether to use RAG
    """
    return DOMAIN_RAG_ENABLED.get(domain, False)
