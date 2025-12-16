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

logger = get_logger(__name__)

# Default configuration (can be overridden by .env)
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 50  # characters
DEFAULT_EMBEDDING_DIM = 768  # Common embedding dimension

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
    # Get chunking parameters from config
    chunk_size = config.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = config.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    embedding_dim = config.get("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    
    all_chunks = []
    logger.info(f"Loading documents from {docs_dir}...")
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.json', '.csv', '.xlsx', '.xls', '.docx', '.doc', '.md', '.txt'}
    tabular_extensions = {'.json', '.csv', '.xlsx', '.xls'}  # Formats needing specialized chunking
    
    # Get list of files to process
    files_to_process = [f for f in os.listdir(docs_dir) if Path(f).suffix.lower() in supported_extensions]
    
    for filename in tqdm(files_to_process, desc="Processing documents", unit="file"):
        file_ext = Path(filename).suffix.lower()
        filepath = os.path.join(docs_dir, filename)
        try:
            # Use specialized chunking for tabular data
            if file_ext in tabular_extensions:
                chunks = chunk_tabular_data(filepath, file_ext, chunk_size)
                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {filename} (tabular): {len(chunks)} chunks.")
            else:
                # Regular text-based chunking for other formats
                text = extract_text_from_file(filepath)
                if text:  # Only process if we got text
                    # Use markdown-aware chunking for .md files
                    is_markdown = (file_ext == '.md')
                    chunks = chunk_text(text, chunk_size, chunk_overlap, is_markdown=is_markdown)
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {filename}: {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    if not all_chunks:
        logger.warning("No text chunks extracted from PDFs. Index will be empty.")

    if not all_chunks:
        logger.warning("No text chunks extracted from PDFs. Index will be empty.")
        return

    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info("Generating embeddings (this may take a while)...")

    embeddings = await generate_embeddings_async(all_chunks, config)

    d = embeddings.shape[1] if embeddings.shape[0] > 0 else embedding_dim
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True) # Create directory if it doesn't exist
    faiss.write_index(index, index_path)
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    logger.info(f"FAISS index built and saved to {index_path}")
    logger.info(f"Text chunks saved to {texts_path}")

    # Build and save BM25 index
    logger.info("Building BM25 index...")
    tokenized_corpus = [ViTokenizer.tokenize(doc).split() for doc in tqdm(all_chunks, desc="Tokenizing for BM25", unit="chunk")]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_index_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index built and saved to {bm25_index_path}")


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
