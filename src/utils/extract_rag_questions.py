import json
import os

def extract_rag_questions():
    # Paths
    # Assuming this script is in src/utils/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    classification_path = os.path.join(base_dir, 'results', 'test_classification.json')
    test_data_path = os.path.join(base_dir, 'data', 'test.json')
    output_path = os.path.join(base_dir, 'results', 'rag_necessity_questions.txt')

    if not os.path.exists(classification_path):
        print(f"Error: File not found: {classification_path}")
        return

    if not os.path.exists(test_data_path):
        print(f"Error: File not found: {test_data_path}")
        return

    print(f"Reading classification from {classification_path}")
    with open(classification_path, 'r', encoding='utf-8') as f:
        classification_data = json.load(f)

    print(f"Reading test data from {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Create a map of qid to question from test data
    qid_to_question = {item['qid']: item['question'] for item in test_data}

    rag_questions = []
    count = 0
    
    for item in classification_data:
        if item.get('predicted_domain') == 'RAG_NECESSITY':
            qid = item.get('qid')
            if qid in qid_to_question:
                rag_questions.append(qid_to_question[qid])
                count += 1
            else:
                print(f"Warning: QID {qid} not found in test data.")
                # Fallback if qid not found (unlikely) or use snippet if available
                if 'question_snippet' in item:
                     rag_questions.append(item['question_snippet'])
                     count += 1

    print(f"Found {count} questions with RAG_NECESSITY.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in rag_questions:
            f.write(q + '\n')
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract_rag_questions()
