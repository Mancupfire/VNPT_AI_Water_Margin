import os
import requests
import aiohttp
from .base import Provider
from typing import Dict, Any, List
import asyncio
import time
from functools import wraps

DEFAULT_EMBEDDING_DIMENSION = 1024

class VNPTProvider:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VNPT_API_BASE", "https://api.idg.vnpt.vn/data-service/v1/chat/completions")
        self.embedding_url = os.getenv("VNPT_EMBEDDING_API_BASE", "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding")
        self._credentials_cache = None  # Cache for JSON credentials
        self.enable_infinite_retry = os.getenv("VNPT_INFINITE_RETRY", "true").lower() == "true"
        self.initial_retry_delay = float(os.getenv("VNPT_RETRY_INITIAL_DELAY", "5"))  # seconds
        self.max_retry_delay = float(os.getenv("VNPT_RETRY_MAX_DELAY", "300"))  # 5 minutes max

    def chat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.achat(messages, config))

    def _load_credentials_from_json(self):
        """Load credentials from .secret/api-keys.json if available."""
        if self._credentials_cache is not None:
            return self._credentials_cache
        
        try:
            import json
            json_path = os.path.join(os.getcwd(), ".secret", "api-keys.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    creds = json.load(f)
                    # Map credentials by model type
                    self._credentials_cache = {
                        'embedding': next((c for c in creds if 'embedding' in c.get('llmApiName', '').lower()), None),
                        'large': next((c for c in creds if 'large' in c.get('llmApiName', '').lower()), None),
                        'small': next((c for c in creds if 'small' in c.get('llmApiName', '').lower()), None),
                    }
                    return self._credentials_cache
        except Exception as e:
            print(f"Warning: Could not load credentials from .secret/api-keys.json: {e}")
        
        self._credentials_cache = {}
        return self._credentials_cache

    def _get_credentials(self, model_name: str, config: Dict[str, Any]):
        """Helper to get the correct Token ID and Key based on the model name."""
        access_token = None
        token_id = None
        token_key = None
        model_type = None

        # Step 1: Check for model-specific environment variables
        if "small" in model_name.lower():
            model_type = "small"
            access_token = os.getenv("VNPT_SMALL_ACCESS_TOKEN")
            token_id = os.getenv("VNPT_SMALL_TOKEN_ID")
            token_key = os.getenv("VNPT_SMALL_TOKEN_KEY")
        elif "large" in model_name.lower():
            model_type = "large"
            access_token = os.getenv("VNPT_LARGE_ACCESS_TOKEN")
            token_id = os.getenv("VNPT_LARGE_TOKEN_ID")
            token_key = os.getenv("VNPT_LARGE_TOKEN_KEY")
        elif "embedding" in model_name.lower():
            model_type = "embedding"
            access_token = os.getenv("VNPT_EMBEDDING_ACCESS_TOKEN")
            token_id = os.getenv("VNPT_EMBEDDING_TOKEN_ID")
            token_key = os.getenv("VNPT_EMBEDDING_TOKEN_KEY")
        
        # Step 2: If not found in env vars, use JSON file as PRIMARY DEFAULT
        using_json = False
        if (not access_token or not token_id or not token_key) and model_type:
            json_creds = self._load_credentials_from_json()
            if model_type in json_creds and json_creds[model_type]:
                cred = json_creds[model_type]
                if not access_token:
                    auth = cred.get('authorization', '')
                    # Remove 'Bearer ' prefix if present
                    access_token = auth.replace('Bearer ', '') if auth.startswith('Bearer ') else auth
                    if access_token:
                        using_json = True
                if not token_id:
                    token_id = cred.get('tokenId')
                if not token_key:
                    token_key = cred.get('tokenKey')
        
        # Step 3: Generic fallback (only if JSON also didn't have credentials)
        using_fallback = False
        if not access_token:
            access_token = config.get("ACCESS_TOKEN") or os.getenv("VNPT_ACCESS_TOKEN")
            if access_token and model_type:
                using_fallback = True
        if not token_id:
            token_id = config.get("TOKEN_ID") or os.getenv("VNPT_TOKEN_ID")
            if token_id and model_type:
                using_fallback = True
        if not token_key:
            token_key = config.get("TOKEN_KEY") or os.getenv("VNPT_TOKEN_KEY")
            if token_key and model_type:
                using_fallback = True
        
        # Info/Warning messages
        if using_json and model_type:
            # Success message when using JSON (only show once per model type)
            cache_key = f"_json_msg_{model_type}"
            if not hasattr(self, cache_key):
                print(f"\nℹ️  Using credentials from .secret/api-keys.json for {model_type} model")
                print(f"   ACCESS_TOKEN: {'✓ Loaded' if access_token else '✗ Missing'}")
                print(f"   TOKEN_ID: {token_id[:20]}... ✓" if token_id else "   TOKEN_ID: ✗ Missing")
                print(f"   TOKEN_KEY: {'✓ Loaded' if token_key else '✗ Missing'}")
                setattr(self, cache_key, True)
        elif using_fallback and model_type:
            # Warning when using generic fallback
            print(f"\n⚠️  Warning: Using generic fallback credentials for {model_type} model. "
                  f"For proper authentication, add credentials to .secret/api-keys.json or set "
                  f"VNPT_{model_type.upper()}_ACCESS_TOKEN, VNPT_{model_type.upper()}_TOKEN_ID, "
                  f"and VNPT_{model_type.upper()}_TOKEN_KEY in .env")
        
        # Validation
        if not access_token:
            print("❌ Error: VNPT_ACCESS_TOKEN not found. Please add credentials to .secret/api-keys.json or .env")
        if not token_id:
            print("❌ Error: VNPT Token ID not found. Please configure credentials.")
        if not token_key:
            print("❌ Error: VNPT Token Key not found. Please configure credentials.")
            
        return access_token, token_id, token_key

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry wrapper with exponential backoff for handling quota limits.
        
        Retries infinitely on quota errors (429) or rate limit errors.
        Uses exponential backoff starting from initial_retry_delay, capped at max_retry_delay.
        """
        retry_count = 0
        delay = self.initial_retry_delay
        
        while True:
            try:
                return await func(*args, **kwargs)
            except aiohttp.ClientResponseError as e:
                # Check if it's a quota/rate limit error
                if e.status == 429 or (e.status >= 500 and e.status < 600) or e.status == 401:
                    if not self.enable_infinite_retry:
                        raise
                    
                    retry_count += 1
                    print(f"\n⚠️  Quota/Rate limit hit (HTTP {e.status}). Retry #{retry_count} after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    
                    # Exponential backoff with cap
                    delay = min(delay * 2, self.max_retry_delay)
                else:
                    print(f"Unexpected error: {str(e)}. Retry #{retry_count} after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.max_retry_delay)
                    
            except aiohttp.ClientError as e:
                # Network/connection errors - retry with same logic
                if not self.enable_infinite_retry:
                    raise
                
                retry_count += 1
                print(f"\n⚠️  Network error: {str(e)}. Retry #{retry_count} after {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_retry_delay)
            except Exception as e:
                # Check if error message contains quota-related keywords
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['quota', 'rate limit', 'too many requests']):
                    if not self.enable_infinite_retry:
                        raise
                    
                    retry_count += 1
                    print(f"\n⚠️  Quota error detected: {str(e)}. Retry #{retry_count} after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.max_retry_delay)
                else:
                    print(f"Unexpected error: {str(e)}. Retry #{retry_count} after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.max_retry_delay)

    async def achat(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Send chat completion request with automatic retry on quota limits."""
        return await self._retry_with_backoff(self._achat_inner, messages, config)
    
    async def _achat_inner(self, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Inner chat method that performs the actual API call."""
        model_name = config.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "vnptai-hackathon-small"
        api_base = self.base_url.rstrip('/')
        url = f"{api_base}/{model_name}"

        headers = {"Content-Type": "application/json"}
        access_token, token_id, token_key = self._get_credentials(model_name, config)

        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if token_id:
            headers["Token-id"] = token_id
        if token_key:
            headers["Token-key"] = token_key

        model_body_name = model_name.replace('-', '_')
        payload = {"model": model_body_name, "messages": messages}
        payload.update(config.get("PAYLOAD_HYPERPARAMS", {}))

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if 'error' in data:
                    error_msg = data['error'].get('message', str(data['error']))
                    # Check if it's a quota error
                    if 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                        raise Exception(f"Quota error: {error_msg}")
                    return f"Error from VNPT API: {error_msg}"
                if isinstance(data, dict) and data.get("choices"):
                    return data["choices"][0]["message"]["content"]
                return ""

    async def aembed(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings with automatic retry on quota limits."""
        return await self._retry_with_backoff(self._aembed_inner, texts, config)
    
    async def _aembed_inner(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Inner embedding method that performs the actual API calls."""
        model_name = config.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "vnptai_hackathon_embedding"
        url = self.embedding_url

        headers = {"Content-Type": "application/json"}
        access_token, token_id, token_key = self._get_credentials(model_name, config)

        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if token_id:
            headers["Token-id"] = token_id
        if token_key:
            headers["Token-key"] = token_key

        all_embeddings = []
        # Create session once and reuse for all requests (performance optimization)
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": model_name,
                    "input": text,
                    "encoding_format": "float"
                }
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        if 'error' in data:
                            error_msg = data['error'].get('message', str(data['error']))
                            # Check if it's a quota error
                            if 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                                raise Exception(f"Quota error: {error_msg}")
                            raise Exception(f"Error from VNPT Embedding API: {error_msg}")
                        if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                            embedding = data["data"][0]["embedding"]
                            # Validate embedding dimension (must be 1024)
                            if len(embedding) != DEFAULT_EMBEDDING_DIMENSION:
                                raise Exception(f"Invalid embedding dimension: expected {DEFAULT_EMBEDDING_DIMENSION}, got {len(embedding)}. Retrying...")
                            all_embeddings.append(embedding)
                        else:
                            all_embeddings.append([])
                except Exception as e:
                    # Check if it's a quota error or dimension error that should be retried
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['quota', 'rate limit', 'too many requests', 'invalid embedding dimension']):
                        raise  # Re-raise to trigger retry
                    # Other errors: log and continue with empty embedding
                    print(f"Warning: Failed to generate embedding for text (length {len(text)}): {str(e)}")
                    all_embeddings.append([])
        return all_embeddings


def create(config: Dict[str, Any] | None = None) -> VNPTProvider:
    return VNPTProvider()
