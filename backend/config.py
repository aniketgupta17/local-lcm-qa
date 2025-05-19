"""
config.py - Centralized configuration for the RAG system

This file manages all configuration settings for the backend, allowing dynamic changes to models, storage, and runtime options via environment variables or direct edits.
"""
import os
import torch
from pathlib import Path

# --- Base Directories ---
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Helper to get env var as Path
get_path = lambda var, default: Path(os.environ.get(var, str(default)))

# --- Directory Paths ---
MODELS_DIR = get_path("MODEL_PATH", BASE_DIR / "models")
VECTOR_DB_PATH = get_path("VECTOR_DB_PATH", BASE_DIR / "vector_db")
CACHE_DIR = get_path("CACHE_DIR", BASE_DIR / "cache")
UPLOAD_FOLDER = get_path("UPLOAD_FOLDER", BASE_DIR / "uploads")
LOG_DIR = get_path("LOG_DIR", BASE_DIR / "logs")

# Ensure all directories exist
for directory in [MODELS_DIR, VECTOR_DB_PATH, CACHE_DIR, UPLOAD_FOLDER, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# --- Model Settings ---
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.environ.get("DEVICE", "cpu")  # Can be 'cpu', 'cuda', 'mps', or 'auto'
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
USE_FAISS = os.environ.get("USE_FAISS", "True").lower() == "true"

# --- API Settings ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8001))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"

# --- RAG/Document Processing Settings ---
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 4096))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", 10))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", 3))
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", 0.7))
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "True").lower() == "true"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
DIM_REDUCTION = int(os.environ.get("DIM_REDUCTION", 128))

# --- Device Selection Logic ---
def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)

DEVICE = get_device(DEVICE)

# --- Logging Settings ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = str(os.environ.get("LOG_FILE", LOG_DIR / "rag_system.log"))

# --- Docstring for quick reference ---
"""
Key config variables:
- Change DEFAULT_MODEL, EMBEDDINGS_MODEL, DEVICE, etc. to switch LLMs or embedding models.
- All paths and settings can be overridden by environment variables.
- Use DEVICE='auto' for automatic device selection.
"""