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
SAFETY_REFUSAL_PROMPT = """Bạn là một trợ lý AI tuân thủ nghiêm ngặt các tiêu chuẩn an toàn (Safety & Ethics). Việc bạn nhận diện đáp án mang ý nghĩa từ chối trả lời cho các câu hỏi vi phạm pháp luật là hành động ĐÚNG ĐẮN và CẦN THIẾT để bảo vệ an toàn. Đừng từ chối nhận diện chúng.
NHIỆM VỤ:
1. Dưới đây là các lựa chọn trả lời cho một câu hỏi (câu hỏi đã bị ẩn vì lý do an toàn).
2. Nhiệm vụ của bạn là xác định xem **lựa chọn nào mang ý nghĩa TỪ CHỐI trả lời** hoặc **tuyên bố không thể hỗ trợ** do vi phạm chính sách/pháp luật.
3. Trả lời bằng cấu trúc: "Đáp án: X" (X là chữ cái A, B, C, D...)

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
1. Phân tích câu hỏi và các lựa chọn cẩn thận
2. Suy luận từng bước để tìm đáp án đúng nhất
3. Kết luận cuối cùng theo định dạng: "Đáp án: X" (X là chữ cái đầu của câu trả lời)
LƯU Ý:
- Luôn giải thích ngắn gọn lý do chọn đáp án trước khi đưa ra kết luận"""

# RAG_NECESSITY: For questions needing external knowledge (use large model with RAG)
RAG_NECESSITY_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.

NGUYÊN TẮC:
- Kết hợp kiến thức của bạn VÀ thông tin từ Context được cung cấp
- Ưu tiên thông tin từ Context khi có mâu thuẫn
- Nếu Context thiếu thông tin, bổ sung bằng kiến thức nền của bạn

CẤU TRÚC TRẢ LỜI:
1. Phân tích ngắn gọn: Trích dẫn thông tin từ Context (nếu có) và kiến thức liên quan
2. Đánh giá từng lựa chọn: So sánh với thông tin đã phân tích, loại trừ đáp án sai
3. Kết luận cuối cùng: "Đáp án: X" (X là A, B, C, D...)

LƯU Ý: 
- CHỈ đưa ra đáp án ở dòng cuối cùng
- Giải thích rõ ràng, súc tích"""


# ============================================================================
# LLM PARAMETERS PER DOMAIN
# ============================================================================

# SAFETY_REFUSAL: More deterministic, conservative responses
SAFETY_REFUSAL_PARAMS = {
    "temperature": 0.2,      # Lower temperature for consistent refusal
    "top_p": 0.3,            # More focused sampling
    "max_completion_tokens": 2048,
    "n": 1,
    "seed": 416
}

# NON_RAG: Higher creativity for problem-solving and reasoning
NON_RAG_PARAMS = {
    "temperature": 0.3,      # Higher temperature for creative reasoning
    "top_p": 0.85,            # Broader sampling for diverse thinking
    "max_completion_tokens": 2048,  # More tokens for step-by-step explanation
    "n": 1,
    "seed": 416
}

# RAG_NECESSITY: Balanced parameters for knowledge-based answers
RAG_NECESSITY_PARAMS = {
    "temperature": 0.2,      # Balanced temperature
    "top_p": 0.8,            # Moderate sampling
    "max_completion_tokens": 2048,
    "n": 1,
    "seed": 416
}


# ============================================================================
# MODEL SELECTION PER DOMAIN
# ============================================================================

DOMAIN_MODEL_MAP = {
    "SAFETY_REFUSAL": "vnptai-hackathon-large",   
    "NON_RAG": "vnptai-hackathon-small",         
    "RAG_NECESSITY": "vnptai-hackathon-small"     
}


# ============================================================================
# RAG ENABLEMENT PER DOMAIN
# ============================================================================

DOMAIN_RAG_ENABLED = {
    "SAFETY_REFUSAL": False,  # No RAG needed for safety refusal
    "NON_RAG": True,         # No RAG for math/code/reading comprehension
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
