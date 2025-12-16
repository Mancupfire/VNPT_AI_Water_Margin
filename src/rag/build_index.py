import os
import sys
import json
import asyncio
from pypdf import PdfReader
import faiss
import numpy as np
from typing import Dict, Any, List
from rank_bm25 import BM25Okapi
import pickle
from pyvi import ViTokenizer
import csv
from pathlib import Path
from tqdm import tqdm

# Add project root to path for direct script execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.providers.factory import load_embedding_provider
from src.logger import get_logger
import hashlib

logger = get_logger(__name__)

# Default configuration (can be overridden by .env)
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 50  # characters
DEFAULT_EMBEDDING_DIM = 768  # Common embedding dimension


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {filepath}: {e}")
        return ""


def load_existing_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load existing metadata tracking processed files."""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {"files": {}, "version": "1.0"}
    return {"files": {}, "version": "1.0"}


def save_metadata(metadata_path: str, metadata: Dict[str, Any]):
    """Save metadata tracking processed files."""
    try:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")


def load_existing_index(index_path: str, texts_path: str, bm25_path: str) -> tuple:
    """
    Load existing FAISS index, text chunks, and BM25 index if they exist.
    Returns (faiss_index, text_chunks, bm25_index) or (None, [], None) if not found.
    """
    faiss_index = None
    text_chunks = []
    bm25_index = None
    
    # Load FAISS index
    if os.path.exists(index_path):
        try:
            faiss_index = faiss.read_index(index_path)
            logger.info(f"✓ Loaded existing FAISS index with {faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    
    # Load text chunks
    if os.path.exists(texts_path):
        try:
            with open(texts_path, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            logger.info(f"✓ Loaded {len(text_chunks)} existing text chunks")
        except Exception as e:
            logger.error(f"Error loading text chunks: {e}")
    
    # Load BM25 index
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, 'rb') as f:
                bm25_index = pickle.load(f)
            logger.info(f"✓ Loaded existing BM25 index")
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
    
    return faiss_index, text_chunks, bm25_index


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, is_markdown: bool = False) -> List[str]:
    """
    Chunk text using LangChain's recursive character splitter.
    For markdown files, uses MarkdownTextSplitter to preserve structure.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        is_markdown: Whether the text is markdown format
        
    Returns:
        List of text chunks
    """
    try:
        if is_markdown:
            # Use MarkdownTextSplitter for markdown files
            from langchain_text_splitters import MarkdownTextSplitter
            
            splitter = MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # Use RecursiveCharacterTextSplitter for all other files
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        
        chunks = splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]
    
    except ImportError:
        logger.warning("langchain not installed, falling back to simple chunking")
        # Fallback to simple chunking if langchain not available
        return _simple_chunk_text(text, chunk_size, chunk_overlap)

def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple fallback chunking method if langchain is not available."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end > text_len:
            end = text_len
        
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        
        start += chunk_size - chunk_overlap
    
    return chunks

def chunk_tabular_data(filepath: str, file_ext: str, chunk_size: int) -> List[str]:
    """
    Specialized chunking for tabular data (JSON, CSV, XLSX).
    Preserves row integrity and data structure.
    
    Args:
        filepath: Path to the file
        file_ext: File extension (.json, .csv, .xlsx, .xls)
        chunk_size: Approximate size of each chunk in characters
        
    Returns:
        List of text chunks, each representing logical groups of rows
    """
    chunks = []
    
    try:
        if file_ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects - chunk by groups of items
                items_per_chunk = max(1, chunk_size // 200)  # Estimate ~200 chars per item
                for i in range(0, len(data), items_per_chunk):
                    chunk_data = data[i:i + items_per_chunk]
                    chunk_text = json.dumps(chunk_data, ensure_ascii=False, indent=2)
                    chunks.append(chunk_text)
            elif isinstance(data, dict):
                # Single object - chunk by top-level keys
                temp_chunk = {}
                current_size = 0
                
                for key, value in data.items():
                    item_text = json.dumps({key: value}, ensure_ascii=False)
                    item_size = len(item_text)
                    
                    if current_size + item_size > chunk_size and temp_chunk:
                        chunks.append(json.dumps(temp_chunk, ensure_ascii=False, indent=2))
                        temp_chunk = {}
                        current_size = 0
                    
                    temp_chunk[key] = value
                    current_size += item_size
                
                if temp_chunk:
                    chunks.append(json.dumps(temp_chunk, ensure_ascii=False, indent=2))
            else:
                # Primitive type - just convert to string
                chunks.append(json.dumps(data, ensure_ascii=False))
        
        elif file_ext == '.csv':
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if not rows:
                return []
            
            # Keep header and chunk data rows
            header = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            
            # Estimate rows per chunk
            avg_row_size = sum(len(" | ".join(row)) for row in rows[:10]) // min(10, len(rows)) if rows else 100
            rows_per_chunk = max(1, chunk_size // avg_row_size)
            
            for i in range(0, len(data_rows), rows_per_chunk):
                chunk_rows = [header] + data_rows[i:i + rows_per_chunk]
                chunk_text = "\n".join([" | ".join(row) for row in chunk_rows])
                chunks.append(chunk_text)
        
        elif file_ext in ['.xlsx', '.xls']:
            try:
                import pandas as pd
                df = pd.read_excel(filepath)
                
                if df.empty:
                    return []
                
                # Estimate rows per chunk based on dataframe size
                avg_row_size = len(df.head(10).to_string()) // min(10, len(df))
                rows_per_chunk = max(1, chunk_size // avg_row_size)
                
                # Chunk by rows, keeping header info
                for i in range(0, len(df), rows_per_chunk):
                    chunk_df = df.iloc[i:i + rows_per_chunk]
                    chunk_text = chunk_df.to_string(index=False)
                    chunks.append(chunk_text)
            
            except ImportError:
                logger.warning(f"pandas not installed, cannot chunk {filepath}")
                return []
    
    except Exception as e:
        logger.error(f"Error chunking tabular data from {filepath}: {e}")
        return []
    
    return [chunk for chunk in chunks if chunk.strip()]

def extract_text_from_file(filepath: str) -> str:
    """
    Extract text from various file formats.
    Supports: PDF, JSON, CSV, XLSX, DOC, DOCX, MD, TXT
    """
    file_ext = Path(filepath).suffix.lower()
    text = ""
    
    try:
        if file_ext == '.pdf':
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        elif file_ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to readable text
                text = json.dumps(data, ensure_ascii=False, indent=2)
        
        elif file_ext == '.csv':
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += " | ".join(row) + "\n"
        
        elif file_ext in ['.xlsx', '.xls']:
            try:
                import pandas as pd
                df = pd.read_excel(filepath)
                text = df.to_string(index=False)
            except ImportError:
                logger.warning(f"pandas not installed, skipping {filepath}")
                return ""
        
        elif file_ext in ['.docx', '.doc']:
            try:
                from docx import Document
                doc = Document(filepath)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            except ImportError:
                logger.warning(f"python-docx not installed, skipping {filepath}")
                return ""
        
        elif file_ext in ['.md', '.txt']:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""
    
    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {e}")
        return ""
    
    return text

async def generate_embeddings_async(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Generate embeddings for a list of texts with progress tracking.
    Processes in batches to show progress and handle large datasets.
    """
    embedding_provider = load_embedding_provider(config)
    embedding_dim = config.get("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    batch_size = config.get("EMBEDDING_BATCH_SIZE", 50)  # Process 50 texts at a time
    
    all_embeddings = []
    
    try:
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = await embedding_provider.aembed(batch, config)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                all_embeddings.extend([np.zeros(embedding_dim) for _ in batch])
        
        return np.array(all_embeddings)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.zeros((len(texts), embedding_dim))  # Return zeros for failed embedding





async def build_index(docs_dir: str, index_path: str, texts_path: str, bm25_index_path: str, config: Dict[str, Any]):
    """
    Build or update RAG indices incrementally.
    Only processes new or modified files, preserves existing index data.
    """
    # Get chunking parameters from config
    chunk_size = config.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = config.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    embedding_dim = config.get("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    
    # Metadata file to track processed files
    metadata_path = os.path.join(os.path.dirname(index_path), "index_metadata.json")
    
    # Load existing metadata and indices
    metadata = load_existing_metadata(metadata_path)
    existing_faiss, existing_chunks, existing_bm25 = load_existing_index(index_path, texts_path, bm25_index_path)
    
    # Track what's new
    new_chunks = []
    new_chunk_sources = []  # Track which file each chunk came from
    files_processed = 0
    files_skipped = 0
    files_updated = 0
    
    logger.info(f"Scanning documents in {docs_dir}...")
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.json', '.csv', '.xlsx', '.xls', '.docx', '.doc', '.md', '.txt'}
    tabular_extensions = {'.json', '.csv', '.xlsx', '.xls'}
    
    # Get list of files to process
    all_files = [f for f in os.listdir(docs_dir) if Path(f).suffix.lower() in supported_extensions]
    
    # Determine which files need processing
    files_to_process = []
    for filename in all_files:
        filepath = os.path.join(docs_dir, filename)
        file_hash = compute_file_hash(filepath)
        
        # Check if file is new or modified
        if filename not in metadata["files"]:
            files_to_process.append((filename, filepath, file_hash, "new"))
        elif metadata["files"][filename].get("hash") != file_hash:
            files_to_process.append((filename, filepath, file_hash, "modified"))
        else:
            files_skipped += 1
    
    if not files_to_process:
        logger.info(f"✓ All {len(all_files)} files already indexed. No new documents to process.")
        return
    
    logger.info(f"📊 Status: {len(files_to_process)} files to process, {files_skipped} files unchanged")
    
    # Process new/modified files
    for filename, filepath, file_hash, status in tqdm(files_to_process, desc="Processing documents", unit="file"):
        file_ext = Path(filename).suffix.lower()
        
        try:
            # Use specialized chunking for tabular data
            if file_ext in tabular_extensions:
                chunks = chunk_tabular_data(filepath, file_ext, chunk_size)
                if chunks:
                    new_chunks.extend(chunks)
                    new_chunk_sources.extend([filename] * len(chunks))
                    logger.info(f"✓ {status.upper()}: {filename} (tabular) → {len(chunks)} chunks")
            else:
                # Regular text-based chunking
                text = extract_text_from_file(filepath)
                if text:
                    is_markdown = (file_ext == '.md')
                    chunks = chunk_text(text, chunk_size, chunk_overlap, is_markdown=is_markdown)
                    new_chunks.extend(chunks)
                    new_chunk_sources.extend([filename] * len(chunks))
                    logger.info(f"✓ {status.upper()}: {filename} → {len(chunks)} chunks")
            
            # Update metadata
            metadata["files"][filename] = {
                "hash": file_hash,
                "processed_at": str(Path(filepath).stat().st_mtime),
                "status": status
            }
            
            if status == "new":
                files_processed += 1
            else:
                files_updated += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
    
    if not new_chunks:
        logger.warning("⚠️  No new text chunks extracted. Index unchanged.")
        return
    
    logger.info(f"📝 Extracted {len(new_chunks)} new chunks from {files_processed + files_updated} files")
    
    # Generate embeddings for new chunks
    logger.info("🔧 Generating embeddings for new chunks...")
    new_embeddings = await generate_embeddings_async(new_chunks, config)
    
    # Merge with existing index or create new
    if existing_faiss is not None and existing_faiss.ntotal > 0:
        logger.info(f"🔄 Merging {len(new_chunks)} new chunks into existing index ({existing_faiss.ntotal} vectors)")
        
        # Add new vectors to existing FAISS index
        existing_faiss.add(new_embeddings)
        final_index = existing_faiss
        
        # Merge text chunks
        final_chunks = existing_chunks + new_chunks
        
    else:
        logger.info(f"🆕 Creating new index with {len(new_chunks)} chunks")
        
        # Create new FAISS index
        d = new_embeddings.shape[1] if new_embeddings.shape[0] > 0 else embedding_dim
        final_index = faiss.IndexFlatL2(d)
        final_index.add(new_embeddings)
        
        final_chunks = new_chunks
    
    # Save updated FAISS index and text chunks
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(final_index, index_path)
    
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False)
    
    logger.info(f"✅ FAISS index updated: {final_index.ntotal} total vectors")
    logger.info(f"✅ Text chunks saved: {len(final_chunks)} total chunks")
    
    # Rebuild BM25 index with all chunks (BM25 doesn't support incremental updates easily)
    logger.info("🔧 Rebuilding BM25 index with all chunks...")
    tokenized_corpus = [ViTokenizer.tokenize(doc).split() for doc in tqdm(final_chunks, desc="Tokenizing for BM25", unit="chunk")]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(bm25_index_path, 'wb') as f:
        pickle.dump(bm25, f)
    
    logger.info(f"✅ BM25 index rebuilt: {len(final_chunks)} documents")
    
    # Save metadata
    save_metadata(metadata_path, metadata)
    logger.info(f"✅ Metadata saved: tracking {len(metadata['files'])} files")
    
    # Summary
    logger.info("=" * 70)
    logger.info("📊 INDEX UPDATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  New files added: {files_processed}")
    logger.info(f"  Files updated: {files_updated}")
    logger.info(f"  Files unchanged: {files_skipped}")
    logger.info(f"  Total indexed files: {len(metadata['files'])}")
    logger.info(f"  Total vectors in FAISS: {final_index.ntotal}")
    logger.info(f"  Total text chunks: {len(final_chunks)}")
    logger.info("=" * 70)



if __name__ == "__main__":
    # Fix logging to support UTF-8 characters (Vietnamese)
    import logging
    import sys
    
    # Force UTF-8 encoding for stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    logger.info("Starting index building process...")
    import dotenv
    dotenv.load_dotenv()  # Load .env variables

    # Build config from environment variables
    config = {
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "vnpt"),
        "HUGGINGFACE_EMBEDDING_MODEL": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "vnptai_hackathon_embedding"),  # Use EMBEDDING_MODEL_NAME for credentials
        
        # RAG chunking parameters
        "CHUNK_SIZE": int(os.getenv("RAG_CHUNK_SIZE", "500")),
        "CHUNK_OVERLAP": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        "EMBEDDING_DIM": int(os.getenv("EMBEDDING_DIM", "768")),
    }

    # Use RETRIEVE_DOCS_DIR from .env, or default to relative path
    docs_directory = os.getenv("RETRIEVE_DOCS_DIR")
    if not docs_directory:
        docs_directory = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
        docs_directory = os.path.abspath(docs_directory)
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base")
    knowledge_base_dir = os.path.abspath(knowledge_base_dir)
    os.makedirs(knowledge_base_dir, exist_ok=True) # Ensure knowledge_base directory exists

    faiss_index_file = os.path.join(knowledge_base_dir, "faiss_index.bin")
    text_chunks_file = os.path.join(knowledge_base_dir, "text_chunks.json")
    bm25_index_file = os.path.join(knowledge_base_dir, "bm25_index.pkl")

    asyncio.run(build_index(docs_directory, faiss_index_file, text_chunks_file, bm25_index_file, config))
    logger.info("Index building process finished.")
