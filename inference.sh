#!/bin/bash
# Inference shell script for Docker submission

set -e  # Exit on error

echo "=========================================="
echo "VNPT AI Water Margin - Inference Pipeline"
echo "=========================================="

# Step 1: Initialize vector database (if needed)
echo "[1/2] Initializing vector database..."
python3 process_data.py

# Step 2: Run inference
echo "[2/2] Running inference on test data..."
python3 inference.py

echo "=========================================="
echo "Inference complete!"
echo "Output files:"
echo "  - submission.csv"
echo "  - submission_time.csv"
echo "=========================================="
