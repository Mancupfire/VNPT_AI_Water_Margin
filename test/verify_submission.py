#!/usr/bin/env python3
"""
Test script to verify Docker submission setup.
Checks all required files and their configuration.
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def check_file(filepath, required=True):
    """Check if file exists and return status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"{GREEN}✓{ENDC} {filepath} ({size:,} bytes)")
        return True
    else:
        if required:
            print(f"{RED}✗{ENDC} {filepath} {RED}[MISSING - REQUIRED]{ENDC}")
        else:
            print(f"{YELLOW}⚠{ENDC} {filepath} {YELLOW}[MISSING - OPTIONAL]{ENDC}")
        return not required

def check_directory(dirpath, required=True):
    """Check if directory exists."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        count = len(os.listdir(dirpath))
        print(f"{GREEN}✓{ENDC} {dirpath}/ ({count} items)")
        return True
    else:
        if required:
            print(f"{RED}✗{ENDC} {dirpath}/ {RED}[MISSING - REQUIRED]{ENDC}")
        else:
            print(f"{YELLOW}⚠{ENDC} {dirpath}/ {YELLOW}[MISSING - OPTIONAL]{ENDC}")
        return not required

def main():
    print("="*70)
    print(f"{BLUE}VNPT AI Water Margin - Docker Submission Verification{ENDC}")
    print("="*70)
    print()
    
    all_ok = True
    
    # Check required submission files
    print(f"{BLUE}[1/5] Required Submission Files{ENDC}")
    print("-"*70)
    all_ok &= check_file("README.md", required=True)
    all_ok &= check_file("requirements.txt", required=True)
    all_ok &= check_file("inference.py", required=True)
    all_ok &= check_file("inference.sh", required=True)
    all_ok &= check_file("Dockerfile", required=True)
    all_ok &= check_file("process_data.py", required=True)
    print()
    
    # Check core application files
    print(f"{BLUE}[2/5] Core Application Files{ENDC}")
    print("-"*70)
    all_ok &= check_file("main.py", required=True)
    all_ok &= check_directory("src", required=True)
    all_ok &= check_directory("src/core", required=True)
    all_ok &= check_directory("src/providers", required=True)
    all_ok &= check_directory("src/rag", required=True)
    all_ok &= check_directory("src/classification", required=True)
    all_ok &= check_directory("src/utils", required=True)
    print()
    
    # Check configuration files
    print(f"{BLUE}[3/5] Configuration Files{ENDC}")
    print("-"*70)
    check_file(".env", required=False)  # Optional, can use env vars
    check_file(".env.example", required=False)
    check_file(".secret/api-keys.json", required=False)  # Will be provided by organizers
    print()
    
    # Check data directories
    print(f"{BLUE}[4/5] Data Directories{ENDC}")
    print("-"*70)
    check_directory("docs", required=False)  # Optional for RAG
    check_directory("data", required=False)  # For local testing
    check_directory("knowledge_base", required=False)  # Will be created during build
    print()
    
    # Check Docker-specific files
    print(f"{BLUE}[5/5] Docker Configuration{ENDC}")
    print("-"*70)
    all_ok &= check_file("Dockerfile", required=True)
    check_file(".dockerignore", required=False)
    check_file("DOCKER_SUBMISSION.md", required=False)
    print()
    
    # Summary
    print("="*70)
    if all_ok:
        print(f"{GREEN}✅ All required files present - Ready for submission!{ENDC}")
        print()
        print(f"{BLUE}Next Steps:{ENDC}")
        print("  1. Build Docker image: docker build -t your_username/vnpt-ai-water-margin:latest .")
        print("  2. Test locally with sample data")
        print("  3. Push to DockerHub: docker push your_username/vnpt-ai-water-margin:latest")
        print("  4. Make GitHub repository public")
        print("  5. Submit via competition portal")
        print()
        print(f"See {YELLOW}DOCKER_SUBMISSION.md{ENDC} for detailed instructions.")
    else:
        print(f"{RED}❌ Some required files are missing!{ENDC}")
        print(f"{YELLOW}Please ensure all required files are present before submission.{ENDC}")
    print("="*70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
