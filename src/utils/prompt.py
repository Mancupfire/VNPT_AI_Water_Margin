"""Prompt formatting utilities for multiple-choice questions."""

from typing import Dict, List, Any, Optional


SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
NHIỆM VỤ:
1. Phân tích câu hỏi và các lựa chọn cẩn thận
2. Suy luận từng bước để tìm đáp án đúng nhất
3. Trả lời theo định dạng: "Đáp án: X" (X là chữ cái đầu của câu trả lời đúng nhất)
LƯU Ý:
- Nếu câu hỏi vi phạm pháp luật/đạo đức, trả lời: "Đáp án: Tôi không thể chia sẻ nội dung liên quan đến vấn đề này"
- Luôn giải thích ngắn gọn lý do chọn đáp án trước khi đưa ra kết luận"""


def format_prompt(item: Dict[str, Any], context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Format a question item into chat messages for the LLM.
    
    Args:
        item: Question dictionary with 'question' and 'choices' keys
        context: Optional RAG context to prepend to the question
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    question = item.get('question', '')
    choices = item.get('choices', [])

    # Generate choice labels (A, B, C, ...)
    labels = [chr(ord('A') + i) for i in range(26)]
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            formatted_choices.append(f"{labels[i]}. {choice}")

    # Build prompt text
    prompt_text = f"{question}\n" + "\n".join(formatted_choices)
    
    # Add context if provided
    if context:
        prompt_text = f"Context: {context}\n\nQuestion: {prompt_text}"

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
