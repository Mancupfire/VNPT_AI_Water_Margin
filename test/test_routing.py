"""
Test script to verify domain-based routing implementation.

This script demonstrates how the routing system works with different question types.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.prompts_config import (
    get_prompt_for_domain,
    get_llm_params_for_domain,
    get_model_for_domain,
    should_use_rag_for_domain
)
from utils.prompt import format_prompt


def test_routing():
    """Test the domain-based routing configuration."""
    
    print("=" * 80)
    print("DOMAIN-BASED ROUTING TEST")
    print("=" * 80)
    
    domains = ["SAFETY_REFUSAL", "NON_RAG", "RAG_NECESSITY"]
    
    # Sample question item
    test_item = {
        "qid": "test_001",
        "question": "Test question?",
        "choices": ["Option A", "Option B", "Option C", "Option D"]
    }
    
    for domain in domains:
        print(f"\n{'=' * 80}")
        print(f"DOMAIN: {domain}")
        print(f"{'=' * 80}")
        
        # Model selection
        model = get_model_for_domain(domain)
        print(f"\n📦 Model: {model}")
        
        # RAG enablement
        use_rag = should_use_rag_for_domain(domain)
        print(f"🔍 Use RAG: {use_rag}")
        
        # LLM parameters
        params = get_llm_params_for_domain(domain)
        print(f"\n⚙️  LLM Parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Prompt
        prompt = get_prompt_for_domain(domain)
        print(f"\n📝 System Prompt (first 150 chars):")
        print(f"   {prompt[:150]}...")
        
        # Format messages
        messages = format_prompt(test_item, context=None, domain=domain)
        print(f"\n💬 Messages structure:")
        print(f"   Role: {messages[0]['role']}")
        print(f"   Content length: {len(messages[0]['content'])} chars")
        print(f"   User message preview: {messages[1]['content'][:80]}...")
    
    print(f"\n{'=' * 80}")
    print("✅ All routing configurations loaded successfully!")
    print(f"{'=' * 80}\n")


def test_with_classification_file():
    """Test with actual classification file if available."""
    
    classification_file = "data/test_classification.json"
    
    if not os.path.exists(classification_file):
        print(f"\n⚠️  Classification file not found: {classification_file}")
        print("   Skipping real data test.\n")
        return
    
    print(f"\n{'=' * 80}")
    print("TESTING WITH ACTUAL CLASSIFICATION DATA")
    print(f"{'=' * 80}\n")
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count domains
    domain_counts = {}
    for item in data:
        domain = item.get('predicted_domain', 'UNKNOWN')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("📊 Domain Distribution:")
    for domain, count in sorted(domain_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"   {domain}: {count} questions ({percentage:.1f}%)")
    
    # Show sample routing for first question of each domain
    print(f"\n{'=' * 80}")
    print("SAMPLE ROUTING FOR EACH DOMAIN")
    print(f"{'=' * 80}")
    
    seen_domains = set()
    for item in data:
        domain = item.get('predicted_domain')
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            
            print(f"\n--- {domain} ---")
            print(f"QID: {item['qid']}")
            print(f"Question: {item['question_snippet']}")
            print(f"Model: {get_model_for_domain(domain)}")
            print(f"RAG: {'Enabled' if should_use_rag_for_domain(domain) else 'Disabled'}")
            print(f"Temperature: {get_llm_params_for_domain(domain)['temperature']}")
            
            if len(seen_domains) == 3:
                break
    
    print(f"\n{'=' * 80}")
    print("✅ Classification data test complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_routing()
    test_with_classification_file()
