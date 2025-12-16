# Domain-Based Routing Configuration

This document explains how to use the domain-based routing feature.

## Quick Start

### Enable/Disable Routing

In your `.env` file:

```bash
# Enable domain-based routing (default)
DOMAIN_ROUTING_ENABLED=true

# Disable domain-based routing (use traditional approach)
DOMAIN_ROUTING_ENABLED=false
```

## How It Works

When `DOMAIN_ROUTING_ENABLED=true`, the system reads the `predicted_domain` field from your classified questions and automatically:

1. **Selects the appropriate model**:
   - `SAFETY_REFUSAL` → `vnptai-hackathon-small`
   - `NON_RAG` → `vnptai-hackathon-large`
   - `RAG_NECESSITY` → `vnptai-hackathon-large`

2. **Applies the optimal prompt**:
   - `SAFETY_REFUSAL` → Standard safety-aware prompt
   - `NON_RAG` → Chain-of-Thought reasoning prompt
   - `RAG_NECESSITY` → Knowledge-focused prompt

3. **Sets optimal LLM parameters**:
   - `SAFETY_REFUSAL` → `temperature=0.3, top_p=0.5`
   - `NON_RAG` → `temperature=0.7, top_p=0.9`
   - `RAG_NECESSITY` → `temperature=0.5, top_p=0.7`

4. **Enables/disables RAG**:
   - `SAFETY_REFUSAL` → RAG disabled
   - `NON_RAG` → RAG disabled
   - `RAG_NECESSITY` → RAG enabled

## When to Enable Routing

✅ **Enable** (`DOMAIN_ROUTING_ENABLED=true`) when:
- You have classified questions with `predicted_domain` field
- You want to optimize model usage (small vs large)
- You want different prompts for different question types
- You want to save costs by using small model for safety questions

❌ **Disable** (`DOMAIN_ROUTING_ENABLED=false`) when:
- You don't have classified questions
- You want to use a single model for all questions
- You want to use custom .env parameters for all questions
- You're testing or debugging

## Configuration Files

### .env Configuration
```bash
# Main toggle
DOMAIN_ROUTING_ENABLED=true

# Fallback model (used when routing is disabled)
MODEL_NAME=vnptai-hackathon-large

# Fallback LLM parameters (used when routing is disabled)
LLM_TEMPERATURE=0.5
LLM_TOP_P=0.7
LLM_MAX_TOKENS=2048
```

### Prompts & Parameters Configuration

Edit `src/utils/prompts_config.py` to customize:

```python
# Change the Chain-of-Thought prompt for NON_RAG
NON_RAG_COT_PROMPT = """Your custom prompt here..."""

# Adjust parameters for SAFETY_REFUSAL
SAFETY_REFUSAL_PARAMS = {
    "temperature": 0.2,  # Your value
    "top_p": 0.4,        # Your value
    ...
}

# Change model selection
DOMAIN_MODEL_MAP = {
    "SAFETY_REFUSAL": "your-small-model",
    "NON_RAG": "your-large-model",
    "RAG_NECESSITY": "your-large-model"
}
```

## Example Usage

### With Routing Enabled

```python
# Input (with predicted_domain)
{
    "qid": "test_001",
    "question": "Calculate 2 + 2",
    "choices": ["3", "4", "5", "6"],
    "predicted_domain": "NON_RAG"
}

# Automatic routing:
# - Uses vnptai-hackathon-large
# - Applies CoT prompt
# - temperature=0.7, top_p=0.9
# - RAG disabled
```

### With Routing Disabled

```python
# Same input
{
    "qid": "test_001",
    "question": "Calculate 2 + 2",
    "choices": ["3", "4", "5", "6"],
    "predicted_domain": "NON_RAG"  # This is ignored
}

# Uses .env settings:
# - Uses MODEL_NAME from .env
# - Uses LLM_TEMPERATURE from .env
# - Uses default prompt
# - RAG_ENABLED from .env
```

## Backward Compatibility

✅ **Completely backward compatible**:
- Default is `DOMAIN_ROUTING_ENABLED=true`
- If `predicted_domain` is missing, defaults to `RAG_NECESSITY`
- Old code without domain field still works
- Can switch between enabled/disabled without code changes

## Troubleshooting

### Routing not working?

1. Check `.env`:
   ```bash
   DOMAIN_ROUTING_ENABLED=true  # Should be true
   ```

2. Check your input data has `predicted_domain`:
   ```json
   {"qid": "...", "predicted_domain": "NON_RAG", ...}
   ```

3. Check console output for model selection messages

### Always uses same model?

- Make sure routing is enabled in `.env`
- Verify `predicted_domain` values are correct (SAFETY_REFUSAL, NON_RAG, or RAG_NECESSITY)
- Check `.secret/api-keys.json` has credentials for both small and large models

### Want to test without routing?

```bash
# In .env
DOMAIN_ROUTING_ENABLED=false
```

Then all questions use your global `.env` settings.
