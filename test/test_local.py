#!/usr/bin/env python3
"""
Simple local test for inference pipeline (without Docker).
Tests with a small subset of questions to verify functionality.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

def test_local_inference():
    """Test inference pipeline locally with small dataset."""
    print("="*70)
    print("LOCAL INFERENCE TEST")
    print("="*70)
    
    # Check if credentials exist
    creds_file = project_root / ".secret" / "api-keys.json"
    if not creds_file.exists():
        print("❌ Credentials not found: .secret/api-keys.json")
        print("   Please add your API credentials to test")
        return False
    
    print("✓ Credentials found")
    
    # Check test data
    test_file = project_root / "data" / "test.json"
    if not test_file.exists():
        print("❌ Test data not found: data/test.json")
        return False
    
    print("✓ Test data found")
    
    # Load small subset
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} questions")
    
    # Create mini test file (first 2 questions only)
    mini_data = data[:2]
    mini_file = project_root / "data" / "mini_test.json"
    
    with open(mini_file, 'w', encoding='utf-8') as f:
        json.dump(mini_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created mini test file with {len(mini_data)} questions")
    
    # Set environment variable
    os.environ["INPUT_FILE"] = str(mini_file)
    os.environ["OUTPUT_DIR"] = str(project_root / "outputs")
    
    # Create outputs directory
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    print()
    print("-"*70)
    print("RUNNING INFERENCE (this may take a minute)...")
    print("-"*70)
    
    try:
        # Import and run inference
        import asyncio
        from inference import run_inference, build_config_from_env
        
        config = build_config_from_env()
        asyncio.run(run_inference(str(mini_file), str(outputs_dir), config))
        
        print()
        print("-"*70)
        print("CHECKING OUTPUTS")
        print("-"*70)
        
        # Check outputs
        submission_file = outputs_dir / "submission.csv"
        submission_time_file = outputs_dir / "submission_time.csv"
        
        if submission_file.exists():
            print("✅ submission.csv created")
            with open(submission_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   Lines: {len(lines)}")
                print(f"   Preview:")
                for line in lines[:4]:
                    print(f"   {line.strip()}")
        else:
            print("❌ submission.csv NOT created")
            return False
        
        if submission_time_file.exists():
            print("✅ submission_time.csv created")
            with open(submission_time_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   Lines: {len(lines)}")
                print(f"   Preview:")
                for line in lines[:4]:
                    print(f"   {line.strip()}")
        else:
            print("❌ submission_time.csv NOT created")
            return False
        
        print()
        print("="*70)
        print("✅ LOCAL TEST PASSED!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Start Docker Desktop")
        print("  2. Run: docker build -t vnpt-ai-water-margin:test .")
        print("  3. Run: docker run --gpus all -v ${PWD}/data/test.json:/code/private_test.json vnpt-ai-water-margin:test")
        print()
        
        return True
        
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_inference()
    sys.exit(0 if success else 1)
