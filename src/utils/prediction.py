"""Prediction parsing and cleaning utilities."""


DEFAULT_ANSWER = "C"


def clean_prediction(pred_text: str) -> str:
    """
    Extract the answer choice (A, B, C, D, etc.) from LLM prediction text.
    
    Uses multiple strategies to find the answer:
    1. Look for "Đáp án:" pattern
    2. Look for "Đáp án đúng:" pattern
    3. Look for "Vậy đáp án là" pattern
    4. Check last few characters for uppercase letter
    5. Return default answer if nothing found
    
    Args:
        pred_text: Raw prediction text from the LLM
        
    Returns:
        Single uppercase letter representing the answer choice
    """
    if not pred_text:
        return DEFAULT_ANSWER
        
    pred_text = str(pred_text).strip()
    
    # Strategy 1: "Đáp án:" pattern
    if "Đáp án:" in pred_text:
        parts = pred_text.split("Đáp án:", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    
    # Strategy 2: "Đáp án đúng:" pattern
    if "Đáp án đúng:" in pred_text:
        parts = pred_text.split("Đáp án đúng:", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    
    # Strategy 3: "Vậy đáp án là" pattern
    if "Vậy đáp án là" in pred_text:
        parts = pred_text.split("Vậy đáp án là", 1)[1]
        for char in parts:
            if char.isalpha() and char.isupper():
                return char
    
    # Strategy 4: Check last few characters
    if len(pred_text) >= 3 and pred_text[-3].isalpha():
        return pred_text[-3].upper()
    if len(pred_text) >= 4 and pred_text[-4].isalpha():
        return pred_text[-4].upper()
    
    # Fallback to default
    return DEFAULT_ANSWER
