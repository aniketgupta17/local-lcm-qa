# app.py - Enhanced Flask Backend
import os
import time
import json
os.environ["TQDM_DISABLE"] = "1"
import torch
import asyncio
import threading
import numpy as np
import faulthandler
faulthandler.enable()
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for handling Cross-Origin Resource Sharing
from werkzeug.utils import secure_filename # For safe file handling
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
import multiprocessing.resource_tracker
import atexit
import dotenv
dotenv.load_dotenv()


def cleanup():
    try:
        multiprocessing.resource_tracker.unregister('/mpx*', 'semaphore')
    except Exception:
        pass

atexit.register(cleanup)

# Import optimized components - Corrected Imports from config
# Ensure ALL necessary variables used from config are listed here
from config import (
    API_HOST,
    API_PORT,
    DEBUG_MODE,
    DEFAULT_MODEL,
    DEVICE,
    VECTOR_DB_PATH,
    EMBEDDINGS_MODEL,
    USE_FAISS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RERANK_TOP_K,
    FINAL_TOP_K,
    USE_HYBRID_SEARCH,
    LOG_FILE,
    LOG_LEVEL,
    UPLOAD_FOLDER,
    BATCH_SIZE,
    MAX_WORKERS,
    CACHE_DIR,
    CONTEXT_SIZE, # Added CONTEXT_SIZE if used in LLMManager
    MODELS_DIR, # Added MODELS_DIR for constructing model path
    DIM_REDUCTION # Added dimension reduction parameter
)
# Assuming all your backend files (config.py, vector_store.py, etc.)
# are in the same directory as app.py or accessible via PYTHONPATH
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from llm_manager import LLMManager
from utils import setup_logging, generate_doc_id, compress_embeddings # Import utils

hf_token = os.getenv("HF_API_TOKEN")
# Setup logging
# Assuming setup_logging is correctly implemented in utils.py and uses config values
logger = setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)

# Ensure upload folder exists (using the Path object from config, converted to string for os.makedirs)
os.makedirs(str(UPLOAD_FOLDER), exist_ok=True)

# Initialize core components
try:
    # Use new VectorStore signature (ChromaDB, local embeddings)
    vector_store = VectorStore(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_model_name=EMBEDDINGS_MODEL,
        collection_name="default_collection"
    )

    llm_manager = LLMManager(
        model_path=DEFAULT_MODEL,
        model_type="huggingface_api",
        device=DEVICE,
        context_size=CONTEXT_SIZE
    )

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_manager=llm_manager,
        rerank_top_k=RERANK_TOP_K,
        final_top_k=FINAL_TOP_K,
        use_hybrid_search=USE_HYBRID_SEARCH
    )

    document_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        cache_dir=str(CACHE_DIR)
    )

except Exception as e:
    logger.error(f"Failed to initialize RAG components: {e}")
    import sys
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)

# Add CORS middleware to allow frontend to connect
CORS(app) # Allows all origins by default. Restrict this in production.

# In-memory document store (consider a persistent store for production)
# Initialize doc_store with some default structure to ensure keys exist early
docs_store: Dict[str, Dict[str, Any]] = {}

# Thread pool for background tasks (using MAX_WORKERS from config)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS) # Use MAX_WORKERS from config


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with system information"""
    system_info = {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "device": str(DEVICE), # Use DEVICE from config
        "embeddings_model": EMBEDDINGS_MODEL,
        "batch_size": BATCH_SIZE,
        "vector_store": "ChromaDB",
        "hybrid_search": USE_HYBRID_SEARCH,  # Added hybrid search status
        "dim_reduction": DIM_REDUCTION,
        "documents_loaded": len(docs_store),                      # New: count of loaded documents
        "requests_processed": rag_pipeline.request_count,         # New: total answer requests handled
        "cache_hits": rag_pipeline.cache_hits,                    # New: cache hit count  # Added dimension reduction info
        "timestamp": time.time()
    }
    return jsonify(system_info)

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all processed documents"""
    docs_list = [
        {
            "id": doc_id,
            "filename": doc.get("filename", "N/A"), # Use .get for safety
            "status": doc.get("status", "unknown"), # Include status
            "processed_at": doc.get("processed_at", "N/A"),
            "chunk_count": len(doc.get("chunks", [])) # Use .get with default empty list
        }
        for doc_id, doc in docs_store.items()
    ]
    return jsonify({"documents": docs_list})

@app.route('/api/documents/<doc_id>', methods=['GET'])
def get_document_details(doc_id):
    """Get detailed information about a document"""
    doc = docs_store.get(doc_id) # Use .get to avoid KeyError
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    # Don't send back embeddings in the response to reduce payload size
    response = {k: v for k, v in doc.items() if k != "compressed_embeddings"}
    return jsonify(response)

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from the system"""
    if doc_id not in docs_store:
        return jsonify({"error": "Document not found."}), 404

    try:
        # Remove from in-memory store
        del docs_store[doc_id]

        # Remove from vector store
        # The enhanced vector_store.delete_collection handles non-existent collections gracefully
        vector_store.delete_collection(doc_id)

        return jsonify({"success": True, "message": f"Document {doc_id} and associated vectors deleted"})
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


def process_document_async(text: str, filename: str, doc_id: str):
    """Background task to process documents asynchronously, with chunk-level summaries."""
    try:
        initial_processed_at = docs_store[doc_id].get("processed_at", time.time())
        docs_store[doc_id].update({
            "status": "processing",
            "processed_at": initial_processed_at,
            "processing_time": 0.0
        })
        start_time = time.time()
        chunks = document_processor.process_text(text, metadata={"source_filename": filename, "doc_id": doc_id})
        if not chunks:
            logger.warning(f"No chunks generated for document {filename} ({doc_id})")
            docs_store[doc_id].update({
                "status": "completed_no_chunks",
                "processing_time": time.time() - start_time
            })
            return
        chunk_texts = [chunk.page_content for chunk in chunks]
        # Summarize each chunk using LLMManager
        chunk_summaries = []
        for chunk in chunks:
            summary = llm_manager.summarize(chunk.page_content, max_tokens=150)
            chunk_summaries.append(summary)
        # Store chunks and metadata
        chunks_data = []
        for i, (chunk, summary) in enumerate(zip(chunks, chunk_summaries)):
            chunk_text = chunk.page_content
            wc = len(chunk_text.split())
            cc = len(chunk_text)
            excerpt = (chunk_text[:200] + '...') if cc > 200 else chunk_text
            chunks_data.append({
                "text": chunk_text,
                "excerpt": excerpt,
                "word_count": wc,
                "char_count": cc,
                "chunk_index": i,
                "summary": summary,
                "metadata": chunk.metadata
            })
        # Add document to vector store (with summaries in metadata)
        chroma_chunks = [
            {"text": chunk["text"], "metadata": {**chunk["metadata"], "summary": chunk["summary"]}} for chunk in chunks_data
        ]
        vector_store.add_documents(doc_id, chroma_chunks)
        processing_time = time.time() - start_time
        docs_store[doc_id].update({
            "filename": filename,
            "chunks": chunks_data,
            "status": "completed",
            "processed_at": initial_processed_at,
            "processing_time": processing_time,
            "doc_id": doc_id,
            "collection_id": doc_id
        })
        logger.info(f"Document {filename} processed successfully in {processing_time:.2f}s. ID: {doc_id}")
    except Exception as e:
        logger.error(f"Error processing document {filename} ({doc_id}): {str(e)}")
        if doc_id in docs_store:
            initial_processed_at = docs_store[doc_id].get("processed_at", time.time())
            docs_store[doc_id].update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - initial_processed_at
            })
        else:
            docs_store[doc_id] = {
                "filename": filename,
                "status": "failed",
                "doc_id": doc_id,
                "processed_at": time.time(),
                "processing_time": 0.0,
                "error": str(e)
            }

def process_file_async(file_path: str, filename: str, doc_id: str):
    """Background task to process files asynchronously, with chunk-level summaries."""
    try:
        initial_processed_at = docs_store[doc_id].get("processed_at", time.time())
        docs_store[doc_id].update({
            "status": "processing",
            "processed_at": initial_processed_at,
            "processing_time": 0.0
        })
        start_time = time.time()
        document_chunks = document_processor.process_file(file_path, use_cache=False)
        logger.info(f"Processed file {filename} into {len(document_chunks)} chunks")
        if not document_chunks:
            logger.warning(f"No chunks generated for file {filename} ({doc_id})")
            docs_store[doc_id].update({
                "status": "completed_no_chunks",
                "processing_time": time.time() - start_time
            })
            return
        chunk_texts = [chunk.page_content for chunk in document_chunks]
        # Summarize each chunk using LLMManager
        chunk_summaries = []
        for chunk in document_chunks:
            summary = llm_manager.summarize(chunk.page_content, max_tokens=150)
            chunk_summaries.append(summary)
        # Store chunks and metadata
        chunks_data = []
        for i, (chunk, summary) in enumerate(zip(document_chunks, chunk_summaries)):
            chunk_text = chunk.page_content
            wc = len(chunk_text.split())
            cc = len(chunk_text)
            excerpt = (chunk_text[:200] + '...') if cc > 200 else chunk_text
            chunks_data.append({
                "text": chunk_text,
                "excerpt": excerpt,
                "word_count": wc,
                "char_count": cc,
                "chunk_index": i,
                "summary": summary,
                "metadata": chunk.metadata
            })
        # Add document to vector store (with summaries in metadata)
        chroma_chunks = [
            {"text": chunk["text"], "metadata": {**chunk["metadata"], "summary": chunk["summary"]}} for chunk in chunks_data
        ]
        vector_store.add_documents(doc_id, chroma_chunks)
        processing_time = time.time() - start_time
        docs_store[doc_id].update({
            "filename": filename,
            "chunks": chunks_data,
            "status": "completed",
            "processed_at": initial_processed_at,
            "processing_time": processing_time,
            "doc_id": doc_id,
            "collection_id": doc_id
        })
        logger.info(f"File {filename} processed successfully in {processing_time:.2f}s. ID: {doc_id}")
    except Exception as e:
        logger.error(f"Error processing file {filename} ({doc_id}): {str(e)}")
        if doc_id in docs_store:
            initial_processed_at = docs_store[doc_id].get("processed_at", time.time())
            docs_store[doc_id].update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - initial_processed_at
            })
        else:
            docs_store[doc_id] = {
                "filename": filename,
                "status": "failed",
                "doc_id": doc_id,
                "processed_at": time.time(),
                "processing_time": 0.0,
                "error": str(e)
            }
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {file_path}: {str(e)}")


@app.route('/api/upload', methods=['POST'])
def upload_document_endpoint():
    """API endpoint to upload and process a document. Returns analytics after processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400
        filename = secure_filename(file.filename)
        raw_content = file.read()
        is_binary = b'\x00' in raw_content[:1024]
        doc_id = generate_doc_id()
        docs_store[doc_id] = {
            "filename": filename,
            "status": "processing",
            "doc_id": doc_id,
            "processed_at": time.time(),
            "processing_time": 0.0,
            "chunks": [],
        }
        if is_binary:
            upload_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{filename}")
            with open(upload_path, 'wb') as f:
                f.write(raw_content)
            logger.info(f"Saved binary file to {upload_path}")
            docs_store[doc_id]['upload_path'] = upload_path
            executor.submit(process_file_async, upload_path, filename, doc_id)
        else:
            try:
                text = raw_content.decode('utf-8')
            except:
                text = raw_content.decode('latin-1')
            executor.submit(process_document_async, text, filename, doc_id)
        # Wait for processing to complete (for demo, in production this should be async)
        import time as _time
        for _ in range(60):
            _time.sleep(1)
            if docs_store[doc_id]["status"] == "completed":
                break
            if docs_store[doc_id]["status"] == "failed":
                return jsonify({'error': docs_store[doc_id].get('error', 'Processing failed.')}), 500
        if docs_store[doc_id]["status"] != "completed":
            return jsonify({'error': 'Processing timed out.'}), 500
        doc_analytics = vector_store.get_document_analytics(doc_id)
        chunk_summaries = [chunk["summary"] for chunk in docs_store[doc_id]["chunks"]]
        return jsonify({
            "document_id": doc_id,
            "filename": filename,
            "status": docs_store[doc_id]["status"],
            "chunk_count": doc_analytics["chunk_count"],
            "chunk_summaries": chunk_summaries,
            "analytics": doc_analytics
        })
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/answer', methods=['POST'])
def answer_endpoint():
    """Answer questions based on document content"""
    try:
        # Parse request data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON.'}), 400

        data = request.json
        doc_id = data.get('document_id') or data.get('collection_id') # Allow both keys
        question = data.get('question', '').strip()
        
        # Check for hybrid search parameter in request
        hybrid_alpha = float(data.get('hybrid_alpha', 0.7))  # Default to 0.7 if not specified

        # Validate inputs
        if not doc_id:
            return jsonify({'error': 'Missing document_id or collection_id'}), 400

        # Check if the document exists and is completed before attempting to answer
        doc = docs_store.get(doc_id)
        if not doc or doc.get("status") != "completed":
            if not doc:
                return jsonify({'error': f'Document "{doc_id}" not found.'}), 404
            elif doc.get("status") == "processing":
                return jsonify({'error': f'Document "{doc_id}" is still processing. Please wait and try again.'}), 409 # Conflict
            elif doc.get("status") == "failed":
                 return jsonify({'error': f'Document "{doc_id}" failed to process. Cannot answer.'}), 500
            else:
                 return jsonify({'error': f'Document "{doc_id}" is not in a completed state.'}), 400


        if not question:
            return jsonify({'error': 'Empty question'}), 400

        # Get LLM settings from request or use defaults
        temperature = float(data.get('temperature', 0.1))
        max_tokens = int(data.get('max_tokens', 300))

        # Use RAG pipeline to get answer with hybrid search parameter
        answer_result = rag_pipeline.generate_answer(
            question=question,
            collection_id=doc_id, # Use doc_id as collection_id
            temperature=temperature,
            max_tokens=max_tokens,
            hybrid_alpha=hybrid_alpha  # Pass hybrid_alpha to the RAG pipeline
        )

        # Get original chunk information for sources if available in docs_store
        sources = []
        # Ensure doc_id is in docs_store and it has 'chunks' key before accessing
        if 'sources' in answer_result and doc_id in docs_store and 'chunks' in docs_store[doc_id]:
            for source in answer_result['sources']:
                chunk_idx = source.get('chunk_index') # Use .get for safety
                # Validate chunk_idx before accessing the list
                if chunk_idx is not None and 0 <= chunk_idx < len(docs_store[doc_id]['chunks']):
                     chunk_data = docs_store[doc_id]['chunks'][chunk_idx]
                     sources.append({
                         'chunk_index': chunk_idx,
                         'summary': chunk_data.get('summary', 'N/A'), # Use .get
                         'excerpt': chunk_data.get('excerpt', 'N/A'), # Use .get
                         'original': chunk_data.get('text', 'N/A'), # Use .get
                         'similarity': source.get('score', 'N/A'), # Use .get
                         'semantic_score': source.get('semantic_score', 'N/A'),  # Add semantic score if available
                         'keyword_score': source.get('keyword_score', 'N/A')  # Add keyword score if available
                     })
                else:
                     logger.warning(f"Invalid chunk index {chunk_idx} found for document {doc_id}. Skipping source.")

        # Prepare response
        response = {
            'answer': answer_result.get('answer', 'Error retrieving answer.'),
            'sources': sources,
            'processing_time': answer_result.get('processing_time', 'N/A'),
            'error': answer_result.get('error'), # Include error if present
            'hybrid_search_used': USE_HYBRID_SEARCH,  # Indicate if hybrid search was used
            'hybrid_alpha': hybrid_alpha  # Include the hybrid alpha value used
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in answer endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/<doc_id>/download', methods=['GET'])
def download_summary(doc_id):
    """Download the document with chunk summaries as a JSON file."""
    doc = docs_store.get(doc_id)
    if not doc:
        return jsonify({'error': f'Document "{doc_id}" not found.'}), 404
    if doc.get("status") != "completed":
        return jsonify({'error': f'Document "{doc_id}" is not ready for download.'}), 409
    # Prepare download data
    download_data = {
        "document_id": doc_id,
        "filename": doc.get("filename"),
        "chunks": [
            {
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "summary": chunk["summary"],
                "metadata": chunk["metadata"]
            }
            for chunk in doc.get("chunks", [])
        ]
    }
    from flask import Response
    return Response(
        json.dumps(download_data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment;filename={doc_id}_chunks_with_summaries.json"}
    )

if __name__ == '__main__':
    # Print startup information
    logger.info(f"Starting server with model: {DEFAULT_MODEL}")
    logger.info(f"Using device: {DEVICE}") # Use DEVICE from config
    logger.info(f"Embeddings model: {EMBEDDINGS_MODEL}")
    logger.info(f"Vector store: ChromaDB with hybrid search: {USE_HYBRID_SEARCH}")
    logger.info(f"Dimension reduction: {DIM_REDUCTION if DIM_REDUCTION else 'None'}")

    # Use host and port from config.py
    app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT, threaded=True)