# VNPT Model-Specific Credentials Guide

## Overview

Each VNPT model type (small, large, embedding) requires **three separate credentials**:
1. **ACCESS_TOKEN** (Bearer token) - Authorization header
2. **TOKEN_ID** - Token-id header  
3. **TOKEN_KEY** - Token-key header

**Important:** All three credentials are **different for each model type**.

## Environment Variable Structure

### Small Model
```dotenv
VNPT_SMALL_ACCESS_TOKEN=eyJhbGciOiJS...
VNPT_SMALL_TOKEN_ID=4525a88b-e7db-4f0c-e063-62199f0a3a11
VNPT_SMALL_TOKEN_KEY=MFwwDQYJKoZIhvc...
```

### Large Model
```dotenv
VNPT_LARGE_ACCESS_TOKEN=eyJhbGciOiJS...
VNPT_LARGE_TOKEN_ID=4525a84b-002f-2031-e063-62199f0af9db
VNPT_LARGE_TOKEN_KEY=MFwwDQYJKoZIhvc...
```

### Embedding Model
```dotenv
VNPT_EMBEDDING_ACCESS_TOKEN=eyJhbGciOiJS...
VNPT_EMBEDDING_TOKEN_ID=4525a842-6ca4-553c-e063-62199f0a1086
VNPT_EMBEDDING_TOKEN_KEY=MFwwDQYJKoZIhvc...
```

### Fallback (Generic)
```dotenv
# Used if model-specific credentials are not set
VNPT_ACCESS_TOKEN=eyJhbGciOiJS...
VNPT_TOKEN_ID=your-generic-token-id
VNPT_TOKEN_KEY=your-generic-token-key
```

## Credential Priority

The provider checks credentials in this order:

1. **Model-specific environment variables** (e.g., `VNPT_SMALL_ACCESS_TOKEN`)
2. **JSON credential file** (`.secret/api-keys.json`)
3. **Fallback generic variables** (`VNPT_ACCESS_TOKEN`)

## Loading from JSON File

If you have credentials in `.secret/api-keys.json`, the provider will automatically use them:

```json
[
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a88b-e7db-4f0c-e063-62199f0a3a11",
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM small"
  },
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a84b-002f-2031-e063-62199f0af9db", 
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM large"
  },
  {
    "authorization": "Bearer eyJhbG...",
    "tokenId": "4525a842-6ca4-553c-e063-62199f0a1086",
    "tokenKey": "MFwwDQYJ...",
    "llmApiName": "LLM embedings"
  }
]
```

## Validation

Use the validation script to check your credentials:

```pwsh
python scripts/validate_credentials.py
```

It will show which credentials are configured for each model.

## Common Issues

### Issue: "Using fallback credentials" warning

**Cause:** Model-specific credentials not set

**Solution:** Set all three model-specific variables:
```dotenv
VNPT_SMALL_ACCESS_TOKEN=...
VNPT_SMALL_TOKEN_ID=...
VNPT_SMALL_TOKEN_KEY=...
```

### Issue: Authentication error for specific model

**Cause:** Wrong ACCESS_TOKEN for that model

**Solution:** Verify each model has its own ACCESS_TOKEN, not shared.

## Migration from Shared Credentials

If you were using the same ACCESS_TOKEN for all models:

**Before:**
```dotenv
VNPT_ACCESS_TOKEN=shared-token
VNPT_SMALL_TOKEN_ID=...
VNPT_LARGE_TOKEN_ID=...
```

**After:**
```dotenv
# Each model gets its own ACCESS_TOKEN
VNPT_SMALL_ACCESS_TOKEN=small-specific-token
VNPT_SMALL_TOKEN_ID=...

VNPT_LARGE_ACCESS_TOKEN=large-specific-token
VNPT_LARGE_TOKEN_ID=...

VNPT_EMBEDDING_ACCESS_TOKEN=embedding-specific-token
VNPT_EMBEDDING_TOKEN_ID=...
```
