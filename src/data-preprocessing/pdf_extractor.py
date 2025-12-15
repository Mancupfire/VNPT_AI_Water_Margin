import re
import os
from pypdf import PdfReader

def is_header_or_footer(line, page_height, y_position=None):
    """
    Identifies if a line is likely a header or footer based on content.
    (Note: pypdf doesn't easily give Y-position, so we use content heuristics here).
    """
    line = line.strip()
    
    # 1. Page numbers (just digits)
    if re.match(r'^\d+$', line):
        return True
        
    # 2. Specific headers found in this book series
    headers = [
        r'LỊCH SỬ VIỆT NAM',
        r'VIỆN HÀN LÂM KHOA HỌC XÃ HỘI VIỆT NAM',
        r'VIỆN SỬ HỌC',
        r'NHÀ XUẤT BẢN KHOA HỌC XÃ HỘI',
        r'TẬP \d+', # Matches TẬP 8, TẬP 10, etc.
        r'CHƯƠNG [IVXLC]+', # We might want to KEEP Chapter headers, so be careful
        r'TRƯỜNG ĐẠI HỌC ĐÀ LẠT',
        r'KHOA NGỮ VĂN VÀ LỊCH SỬ',
        r'GIÁO TRÌNH',
        r'CƠ SỞ VĂN HÓA VIỆT NAM',
        r'Lưu hành nội bộ',
        r'Lâm Đồng, năm \d+',
        r'Biên mục trên xuất bản phẩm',
    ]
    
    # Remove the book title headers but keep Chapter headers
    for pattern in headers:
        if re.search(pattern, line, re.IGNORECASE) and not line.startswith("CHƯƠNG"):
            return True
            
    return False

def clean_and_merge_text(full_text):
    """
    Merges segmented lines into paragraphs and cleans noise.
    """
    lines = full_text.split('\n')
    cleaned_lines = []
    
    # 1. Filter out headers/footers/noise lines
    for line in lines:
        # Remove TOC lines with many dots
        if re.search(r'\.{5,}', line):
            continue
            
        if not is_header_or_footer(line, 0):
            cleaned_lines.append(line.strip())
            
    # 2. Merge lines into paragraphs
    # Heuristic: If a line ends with terminal punctuation (. ! ?), it's likely a paragraph end.
    # If not, it should be merged with the next line.
    merged_text = ""
    for i, line in enumerate(cleaned_lines):
        if not line: continue
        
        # Check if it looks like a footnote (starts with digit-dot at start of line)
        # e.g., "1. Hồ Chí Minh toàn tập..."
        if re.match(r'^\d+\.\s', line):
            # Option: Skip footnotes or put them at the end. Here we skip to reduce noise.
            continue
            
        merged_text += line
        
        # Decisions on whether to insert a space or a newline
        if i < len(cleaned_lines) - 1:
            next_line = cleaned_lines[i+1]
            
            # If current line ends with terminal punctuation, add paragraph break
            if re.search(r'[.!?։]$', line):
                merged_text += "\n\n"
            # If next line starts with a Capital letter (and current doesn't end in punctuation),
            # it might still be a continuation (Vietnamese names), so we generally join with space
            # unless the current line is very short (likely a title).
            elif len(line) < 50 and line.isupper(): # Likely a section title
                merged_text += "\n\n"
            else:
                merged_text += " "
    
    # 3. Final cleanup of spacing
    merged_text = re.sub(r' +', ' ', merged_text) # Fix multiple spaces
    merged_text = re.sub(r'\n\s+', '\n', merged_text) # Fix spaces at start of lines
    
    return merged_text

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        full_text_list = []
        
        # Extract text page by page
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text_list.append(text)
        
        return "\n".join(full_text_list)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def main():
    input_dir = "data/raw-data"
    output_dir = "data/curated-data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all PDFs in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            print(f"Processing {filename}...")
            raw_text = extract_text_from_pdf(pdf_path)
            
            if raw_text:
                final_text = clean_and_merge_text(raw_text)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                print(f"Saved: {txt_filename}")

if __name__ == "__main__":
    main()