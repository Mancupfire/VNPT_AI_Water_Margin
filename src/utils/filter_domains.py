import json
import csv
import os

def filter_and_export(input_file, output_json, output_csv):
    print(f"--- Đang xử lý file: {input_file} ---")
    
    # 1. Đọc dữ liệu đầu vào
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file đầu vào.")
        return

    # 2. Định nghĩa các domain muốn LOẠI BỎ
    # Bro có thể thêm bớt tùy ý ở đây
    EXCLUDED_DOMAINS = [
        "MATH_LOGIC",
        "READING_COMPREHENSION",
        "VN_CORE_KNOWLEDGE",
        "GENERAL_DOMAIN",
        # "SAFETY_REFUSAL"
    ]

    # 3. Lọc dữ liệu
    filtered_data = []
    removed_count = 0
    
    for item in data:
        domain = item.get('domain', 'UNKNOWN')
        
        # Logic lọc: Giữ lại nếu domain KHÔNG nằm trong danh sách loại bỏ
        if domain not in EXCLUDED_DOMAINS:
            filtered_data.append(item)
        else:
            removed_count += 1

    # 4. Lưu file JSON mới (để bro kiểm tra lại reasoning nếu cần)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # 5. Lưu file CSV (Format nộp bài: qid, answer)
    # Hàm làm sạch đáp án nhỏ để đảm bảo chỉ lấy A, B, C, D
    def clean_answer(ans):
        if not ans: return "C" # Mặc định nếu rỗng
        ans = str(ans).strip().upper()
        # Nếu dài quá (vd: "ĐÁP ÁN A"), chỉ lấy ký tự đầu hoặc cuối
        if len(ans) > 1:
            import re
            match = re.search(r'\b([A-F])\b', ans)
            return match.group(1) if match else "C"
        return ans

    csv_rows = []
    for item in filtered_data:
        csv_rows.append({
            'qid': item.get('qid'),
            'answer': clean_answer(item.get('llm_prediction', ''))
        })

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(csv_rows)

    # 6. Báo cáo kết quả
    print("-" * 30)
    print(f"Tổng số câu ban đầu: {len(data)}")
    print(f"Số câu đã loại bỏ (Math/Reading): {removed_count}")
    print(f"Số câu giữ lại (VnCore/General/Safety): {len(filtered_data)}")
    print("-" * 30)
    print(f"Đã lưu JSON lọc tại: {output_json}")
    print(f"Đã lưu CSV nộp bài tại: {output_csv}")

# --- CHẠY CODE ---
if __name__ == "__main__":
    # Bro nhớ đổi tên file input cho đúng với file của bro nhé
    filter_and_export(
        input_file='results/test_answers.json', 
        output_json='results/test_answers_filtered_safety_refusal.json',
        output_csv='results/submission_filtered_safety_refusal.csv'
    )