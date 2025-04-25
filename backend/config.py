# config.py - Configuration settings for RAG system
import os
import torch
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Get directory paths, ensuring they are Path objects
# Use Path() to convert environment variable strings to Path objects
MODELS_DIR = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models"))
VECTOR_DB_PATH = Path(os.environ.get("VECTOR_DB_PATH", BASE_DIR / "vector_db"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", BASE_DIR / "cache"))
UPLOAD_FOLDER = Path(os.environ.get("UPLOAD_FOLDER", BASE_DIR / "uploads"))
LOG_DIR = Path(os.environ.get("LOG_DIR", BASE_DIR / "logs")) # Also include LOG_DIR from env if set

# Create directories if they don't exist
for directory in [MODELS_DIR, VECTOR_DB_PATH, CACHE_DIR, UPLOAD_FOLDER, LOG_DIR]:
    # Ensure each item in the list is treated as a Path object
    Path(directory).mkdir(exist_ok=True, parents=True)


# Model settings
# Using a relative path as default is generally better if the model is in the project structure
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.environ.get("DEVICE", "auto")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
USE_FAISS = os.environ.get("USE_FAISS", "True").lower() == "true"
# MAX_WORKERS defined below under RAG settings - Removed duplicate definition here

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8001))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"

# RAG settings
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 4096))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", 10))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", 3))
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", 0.7))
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "True").lower() == "true"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4)) # Keep this definition


# Determine the device to use
# This block is also in app.py; ensure consistency or use one source
# If both have it, app.py's determination will overwrite config's at runtime
# Keeping it here for completeness based on your original config, but be mindful
# of where the final device is determined when app.py runs.
if DEVICE == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(DEVICE)

# Logging settings
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# Ensure LOG_FILE is also treated as a Path object if it comes from environment variable
LOG_FILE = os.environ.get("LOG_FILE", str(LOG_DIR / "rag_system.log"))
# The logging setup in utils.py expects a string for log_file, so we keep it as string here
# If utils.setup_logging needs a Path object, convert it there.