import json
import csv
import re

def clean_answer(prediction_text):
    """
    Hàm làm sạch câu trả lời.
    Nếu text rác (ví dụ: 'Đáp', 'Câu'), sẽ cố tìm ký tự hoặc mặc định là 'A'.
    """
    if not prediction_text:
        return "A" # Mặc định nếu rỗng
        
    text = prediction_text.strip().upper()
    
    # 1. Nếu nó đã là 1 ký tự chuẩn A-Z thì giữ nguyên
    if len(text) == 1 and text in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text
        
    # 2. Nếu nó dài (ví dụ: "ĐÁP ÁN A"), dùng Regex để bắt ký tự A-Z
    match = re.search(r'\b([A-Z])\b', text)
    if match:
        return match.group(1)
        
    # 3. Trường hợp 'đường cùng' (như chữ 'ĐÁP', 'CÂU' mà không có A/B/C)
    # Ta chọn đại A để file không bị lỗi format khi nộp
    return "A"

def convert_json_to_submission(input_json, output_csv):
    print(f"--- Đang chuyển đổi {input_json} sang {output_csv} ---")
    
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file JSON đầu vào.")
        return

    # Chuẩn bị list kết quả
    rows = []
    
    # Đếm số lượng sửa lỗi
    fixed_count = 0
    
    for item in data:
        qid = item.get('qid')
        raw_pred = item.get('llm_prediction', '')
        
        # Làm sạch đáp án
        final_ans = clean_answer(raw_pred)
        
        # Kiểm tra xem có bị thay đổi không để log ra (debug)
        if final_ans != raw_pred.strip().upper():
            # Chỉ in vài cái đầu để check
            if fixed_count < 5: 
                print(f"Fixed {qid}: '{raw_pred}' -> '{final_ans}'")
            fixed_count += 1
            
        rows.append({'qid': qid, 'answer': final_ans})

    # Ghi ra file CSV
    headers = ['qid', 'answer']
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"--- Hoàn tất! Đã xử lý {len(rows)} dòng. ---")
    print(f"Số câu trả lời đã được tự động sửa lỗi format: {fixed_count}")
    print(f"File kết quả: {output_csv}")

# --- CHẠY CODE ---
if __name__ == "__main__":
    convert_json_to_submission(
        input_json='results/test_answers.json', 
        output_csv='results/test_predictions_final.csv'
    )