#!/bin/bash

# Testing script
# Sử dụng: bash scripts/test.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo Python path: $PYTHONPATH

set -e

python -m src.language_models.call_llm