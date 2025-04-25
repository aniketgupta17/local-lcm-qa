# utils.py - Utility functions for RAG system
import os
import logging
import hashlib
import time
import numpy as np
from typing import List, Dict, Any
import random
import string
import zlib
import base64
import json

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        
    Returns:
        Configured logger
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Basic configuration
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Apply configuration
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    logger = logging.getLogger("rag_system")
    logger.info(f"Logging initialized at {log_level} level")
    
    return logger

def generate_doc_id(prefix: str = "doc") -> str:
    """
    Generate a unique document ID
    
    Args:
        prefix: Prefix for the document ID
        
    Returns:
        Unique document ID
    """
    # Generate random string
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    # Add timestamp for uniqueness
    timestamp = int(time.time())
    
    # Combine to form the ID
    doc_id = f"{prefix}_{timestamp}_{random_str}"
    
    return doc_id

def compress_embeddings(embeddings: List[np.ndarray]) -> str:
    """
    Compress embeddings for efficient storage
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Compressed embeddings as base64 string
    """
    # Convert to numpy array and flatten
    if not embeddings:
        return ""
    
    try:
        # Convert to float16 for compression
        embeddings_array = np.array(embeddings, dtype=np.float16)
        
        # Serialize with numpy
        serialized = embeddings_array.tobytes()
        
        # Compress with zlib
        compressed = zlib.compress(serialized)
        
        # Encode to base64 for storage
        base64_str = base64.b64encode(compressed).decode('ascii')
        
        # Return compressed string
        return base64_str
    except Exception as e:
        logging.error(f"Error compressing embeddings: {str(e)}")
        return ""

def decompress_embeddings(compressed_str: str) -> List[np.ndarray]:
    """
    Decompress embeddings from storage
    
    Args:
        compressed_str: Compressed embeddings as base64 string
        
    Returns:
        List of embedding vectors
    """
    if not compressed_str:
        return []
    
    try:
        # Decode from base64
        compressed = base64.b64decode(compressed_str)
        
        # Decompress with zlib
        serialized = zlib.decompress(compressed)
        
        # Deserialize with numpy
        embeddings_array = np.frombuffer(serialized, dtype=np.float16)
        
        # Reshape based on dimensionality (assuming standard embedding size)
        # This needs to match your vector_store's embedding dimensionality
        embedding_dim = 384  # Change this if your embeddings have different dimension
        num_embeddings = len(embeddings_array) // embedding_dim
        embeddings = embeddings_array.reshape(num_embeddings, embedding_dim)
        
        # Convert to list of arrays
        return [np.array(emb, dtype=np.float32) for emb in embeddings]
    except Exception as e:
        logging.error(f"Error decompressing embeddings: {str(e)}")
        return []

def hash_text(text: str) -> str:
    """
    Generate a hash for text content
    
    Args:
        text: Text to hash
        
    Returns:
        MD5 hash of text
    """
    return hashlib.md5(text.encode()).hexdigest()

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def prepare_response(data: Dict[str, Any], status_code: int = 200) -> Dict[str, Any]:
    """
    Prepare standardized API response
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "data": data,
        "status": "success" if status_code < 400 else "error",
        "timestamp": time.time()
    }
    return response

def count_tokens(text: str, model_type: str = "llama") -> int:
    """
    Approximate token count for a given text
    
    Args:
        text: Input text
        model_type: Model type for tokenization rules
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    
    # Simple approximation - actual tokenization would be more accurate
    # but this is faster for estimation purposes
    if model_type.lower() in ["llama", "llama.cpp"]:
        # Average of 4 chars per token for LLaMA models
        return len(text) // 4
    else:
        # Default to 4 chars per token
        return len(text) // 4