# Quick Start: Using api-keys.json

## Zero Configuration Setup

With `.secret/api-keys.json` present, you **don't need to set any credentials in `.env`**! 

The provider automatically loads credentials from the JSON file.

## How It Works

**Credential Priority:**
1. ✅ Model-specific environment variables (if set in `.env`)
2. ✅ **`.secret/api-keys.json`** ← **PRIMARY DEFAULT** 
3. ⚠️ Generic fallback env vars (last resort)

## Usage

### 1. Ensure JSON File Exists

Your `.secret/api-keys.json` is already configured:
```json
[
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a88b-...",
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM small"
  },
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a84b-...",
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM large"
  },
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a842-...",
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM embedings"
  }
]
```

### 2. Minimal .env Configuration

You only need these in your `.env`:

```dotenv
# Provider selection
CHAT_PROVIDER=vnpt
EMBEDDING_PROVIDER=vnpt
MODEL_NAME=vnptai-hackathon-small

# Retry settings
VNPT_INFINITE_RETRY=true
SLEEP_TIME=0

# RAG settings (if using)
RAG_ENABLED=false

# Logging
LOG_LEVEL=INFO
```

**No credential duplication needed!** ✨

### 3. Run Your Code

```pwsh
python main.py
```

**Output when using JSON:**
```
ℹ️  Using credentials from .secret/api-keys.json for small model
Processing...
```

## Benefits

✅ **Single source of truth** - Credentials only in JSON file  
✅ **No duplication** - Don't copy values to `.env`  
✅ **Easy updates** - Change JSON file only  
✅ **Less error-prone** - No manual copying mistakes  

## When to Use Environment Variables

Only set model-specific env vars if you want to **override** JSON credentials:

```dotenv
# Override only for small model (large and embedding still use JSON)
VNPT_SMALL_ACCESS_TOKEN=different-token
VNPT_SMALL_TOKEN_ID=different-id
VNPT_SMALL_TOKEN_KEY=different-key
```

## Verification

The provider shows which source it's using:

```
ℹ️  Using credentials from .secret/api-keys.json for small model  ← JSON
⚠️  Warning: Using generic fallback credentials for large model    ← Fallback (problem!)
```

If you see the warning, check your JSON file structure.
