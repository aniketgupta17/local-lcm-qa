# app.py - Enhanced Flask Backend
import os
import time
import json
import torch
import asyncio
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for handling Cross-Origin Resource Sharing
from werkzeug.utils import secure_filename # For safe file handling
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

# Import optimized components - Corrected Imports
# Assuming all your backend files (config.py, vector_store.py, etc.)
# are in the same directory as app.py or accessible via PYTHONPATH
from config import (
    API_HOST, API_PORT, DEBUG_MODE,
    DEFAULT_MODEL, DEVICE, VECTOR_DB_PATH,
    EMBEDDINGS_MODEL, USE_FAISS, CHUNK_SIZE,
    CHUNK_OVERLAP, RERANK_TOP_K, FINAL_TOP_K,
    USE_HYBRID_SEARCH, LOG_FILE, LOG_LEVEL,
    UPLOAD_FOLDER,BATCH_SIZE, # ADDED BATCH_SIZE to the import list
    MAX_WORKERS ,CACHE_DIR
)
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from llm_manager import LLMManager
from utils import setup_logging, generate_doc_id, compress_embeddings # Import utils


# Setup logging
# Assuming setup_logging is correctly implemented in utils.py and uses config values
logger = setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize core components
try:
    # Determine the device to use
    # This block was moved from global scope into the try block
    # to ensure DEVICE is set before other components are initialized
    if DEVICE == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)

    # Pass configuration values from config.py
    vector_store = VectorStore(
        model_name=EMBEDDINGS_MODEL,
        vector_db_path=VECTOR_DB_PATH,
        device=device, # Use the determined device
        use_faiss=USE_FAISS,
        dim_reduction=384,  # PCA dimension reduction for better performance (matches default in vector_store)
        batch_size=BATCH_SIZE # Use BATCH_SIZE from config
    )

    # Ensure the model_path is correctly constructed using MODEL_PATH from config
    # Assuming MODEL_PATH is correctly set in your .env or config
    model_directory = os.environ.get("MODEL_PATH", "./models") # Get directory from config/env
    model_full_path = os.path.join(model_directory, DEFAULT_MODEL) # Construct full path
    llm_manager = LLMManager(
        model_path=model_full_path,
        model_type="llama.cpp", # Assuming you are using llama.cpp based on model name
        device=device, # Use the determined device
        context_size=int(os.environ.get("CONTEXT_SIZE", 4096)) # Use CONTEXT_SIZE from config
    )

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_manager=llm_manager,
        rerank_top_k=RERANK_TOP_K, # Use RERANK_TOP_K from config
        final_top_k=FINAL_TOP_K,    # Use FINAL_TOP_K from config
        use_hybrid_search=USE_HYBRID_SEARCH # Use USE_HYBRID_SEARCH from config
    )

    document_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE, # Use CHUNK_SIZE from config
        chunk_overlap=CHUNK_OVERLAP, # Use CHUNK_OVERLAP from config
        cache_dir=CACHE_DIR # Use CACHE_DIR from config
    )

except Exception as e:
    logger.error(f"Failed to initialize RAG components: {e}")
    # Depending on severity, you might want to exit or handle gracefully
    import sys
    sys.exit(1) # Exit if core components fail to initialize

# Initialize Flask app
app = Flask(__name__)

# Add CORS middleware to allow frontend to connect
CORS(app) # Allows all origins by default. Restrict this in production.

# In-memory document store (consider a persistent store for production)
# Initialize doc_store with some default structure to ensure keys exist early
docs_store: Dict[str, Dict[str, Any]] = {}

# Thread pool for background tasks (using MAX_WORKERS from config)
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", 4)))


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with system information"""
    system_info = {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "device": str(device), # Use the locally determined device variable
        "embeddings_model": EMBEDDINGS_MODEL,
        "batch_size": BATCH_SIZE,
        "vector_store": "FAISS" if USE_FAISS else "In-memory",
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
        # Assuming vector_store.delete_collection handles non-existent collections gracefully
        vector_store.delete_collection(doc_id)

        return jsonify({"success": True, "message": f"Document {doc_id} and associated vectors deleted"})
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


def process_document_async(text: str, filename: str, doc_id: str):
    """Background task to process documents asynchronously"""
    try:
        # Ensure initial status and processing time are set immediately
        docs_store[doc_id].update({
            "status": "processing",
            "processed_at": time.time(),
            "processing_time": 0.0 # Initialize processing_time
        })
        start_time = time.time() # Start timing after initial status update

        # Process document text into chunks
        chunks = document_processor.process_text(text, metadata={"source_filename": filename, "doc_id": doc_id})

        if not chunks:
             logger.warning(f"No chunks generated for document {filename} ({doc_id})")
             docs_store[doc_id].update({ # Use update
                 "status": "completed_no_chunks",
                 "processing_time": time.time() - start_time # Update processing time even if no chunks
            })
             return

        # Extract texts from chunks
        chunk_texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings for each chunk
        chunk_embeddings = vector_store.embed_documents(chunk_texts)

        # Store chunks and metadata
        chunks_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
               chunk_text = chunk.page_content
               wc = len(chunk_text.split())
               cc = len(chunk_text)
               excerpt = (chunk_text[:200] + '...') if cc > 200 else chunk_text

               chunks_data.append({
                   "text": chunk_text,
                   # "summary": summary, # Summaries are not generated here currently
                   "excerpt": excerpt,
                   "word_count": wc,
                   "char_count": cc,
                   "chunk_index": i,
                   "metadata": chunk.metadata
               })

        # Add document to vector store
        vector_store.add_documents(doc_id, chunk_texts, chunk_embeddings)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update document info in memory upon completion
        docs_store[doc_id].update({ # Use update
            "filename": filename,
            "chunks": chunks_data,
            "status": "completed", # Mark as completed
            "processed_at": time.time(), # Update completion time
            "processing_time": processing_time, # Final processing time
            "compressed_embeddings": compress_embeddings(chunk_embeddings.tolist()),  # Compress for storage
            "doc_id": doc_id, # Ensure doc_id is stored
            "collection_id": doc_id # Store the collection_id which is the same as doc_id here
        })

        logger.info(f"Document {filename} processed successfully in {processing_time:.2f}s. ID: {doc_id}")

    except Exception as e:
        logger.error(f"Error processing document {filename} ({doc_id}): {str(e)}")
        # Update status to failed in case of error
        if doc_id in docs_store:
            docs_store[doc_id].update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - docs_store[doc_id].get("processed_at", time.time()) # Calculate elapsed time
            })
        else:
            # If it failed before even being added to docs_store (less likely now)
            docs_store[doc_id] = {
                "filename": filename,
                "status": "failed",
                "doc_id": doc_id,
                "processed_at": time.time(),
                "processing_time": 0.0, # Default if start time wasn't set
                "error": str(e)
            }

def process_file_async(file_path: str, filename: str, doc_id: str):
     """Background task to process files asynchronously."""
     try:
          # Ensure initial status and processing time are set immediately
          docs_store[doc_id].update({
              "status": "processing",
              "processed_at": time.time(),
              "processing_time": 0.0 # Initialize processing_time
          })
          start_time = time.time() # Start timing after initial status update

          # Process the file using DocumentProcessor
          # use_cache=False is good for new uploads
          document_chunks = document_processor.process_file(file_path, use_cache=False)
          logger.info(f"Processed file {filename} into {len(document_chunks)} chunks")

          if not document_chunks:
              logger.warning(f"No chunks generated for file {filename} ({doc_id})")
              docs_store[doc_id].update({ # Use update
                  "status": "completed_no_chunks",
                  "processing_time": time.time() - start_time # Update processing time
             })
              return # Exit if no chunks

          # Extract texts from chunks
          chunk_texts = [chunk.page_content for chunk in document_chunks]

          # Generate embeddings for each chunk
          chunk_embeddings = vector_store.embed_documents(chunk_texts)

          # Store chunks and metadata
          chunks_data = []
          for i, (chunk, embedding) in enumerate(zip(document_chunks, chunk_embeddings)):
               chunk_text = chunk.page_content
               wc = len(chunk_text.split())
               cc = len(chunk_text)
               excerpt = (chunk_text[:200] + '...') if cc > 200 else chunk_text

               chunks_data.append({
                   "text": chunk_text,
                   # "summary": summary, # Summaries are not generated here currently
                   "excerpt": excerpt,
                   "word_count": wc,
                   "char_count": cc,
                   "chunk_index": i,
                   "metadata": chunk.metadata
               })

          # Add document to vector store
          vector_store.add_documents(doc_id, chunk_texts, chunk_embeddings)

          # Calculate processing time
          processing_time = time.time() - start_time

          # Update document info in memory upon completion
          docs_store[doc_id].update({
              "filename": filename,
              "chunks": chunks_data,
              "status": "completed", # Mark as completed
              "processed_at": time.time(), # Update completion time
              "processing_time": processing_time, # Final processing time
              "compressed_embeddings": compress_embeddings(chunk_embeddings.tolist()), # Compress for storage
              "doc_id": doc_id,
              "collection_id": doc_id
          })

          logger.info(f"File {filename} processed successfully in {processing_time:.2f}s. ID: {doc_id}")

     except Exception as e:
          logger.error(f"Error processing file {file_path} ({doc_id}): {str(e)}")
          # Update status to failed in case of error
          if doc_id in docs_store:
               docs_store[doc_id].update({
                   "status": "failed",
                   "error": str(e),
                   "processing_time": time.time() - docs_store[doc_id].get("processed_at", time.time()) # Calculate elapsed time
               })
          else:
               # If it failed before even being added to docs_store
               docs_store[doc_id] = {
                   "filename": filename,
                   "status": "failed",
                   "doc_id": doc_id,
                   "processed_at": time.time(),
                   "processing_time": 0.0, # Default if start time wasn't set
                   "error": str(e)
               }
     finally:
          # Clean up the temporary file
          if os.path.exists(file_path):
               os.remove(file_path)
               logger.info(f"Cleaned up temporary file {file_path}")


@app.route('/api/upload', methods=['POST'])
def upload_document_endpoint():
    """API endpoint to upload and process a document."""
    try:
        # Get file input from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        filename = secure_filename(file.filename)
        raw_content = file.read()

        # Determine if the file is binary (like PDF, Docx) or text
        # This is a simplified check; a more robust check might be needed
        # Read a small chunk to check for null bytes
        is_binary = b'\x00' in raw_content[:1024] # Check first 1KB

        # Generate document ID immediately
        doc_id = generate_doc_id()

        # Store initial document info with 'processing' status
        # Initialize keys that will be updated later to prevent KeyErrors
        docs_store[doc_id] = {
            "filename": filename,
            "status": "processing", # Set initial status
            "doc_id": doc_id,
            "processed_at": time.time(), # Timestamp for initiation
            "processing_time": 0.0, # Initialize with a default value
            "chunks": [], # Initialize chunks list
            "compressed_embeddings": "" # Initialize embeddings string
            # Add other expected keys with default values as needed
        }


        if is_binary:
             # Save binary file to disk for processing by DocumentProcessor
             # Include doc_id in filename to link temporary file to the doc_id
             upload_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{filename}")
             with open(upload_path, 'wb') as f:
                  f.write(raw_content)
             logger.info(f"Saved binary file to {upload_path}")

             # Store upload path in docs_store for the async task
             docs_store[doc_id]['upload_path'] = upload_path

             # Submit file processing to the executor
             executor.submit(process_file_async, upload_path, filename, doc_id)

             # Return immediate response
             return jsonify({
                 "document_id": doc_id,
                 "file_info": {"filename": filename},
                 "status": "processing",
                 "message": "File upload received. Processing started in background."
             })

        else:
            # Handle text content directly
            try:
                text = raw_content.decode('utf-8')
            except:
                text = raw_content.decode('latin-1') # Fallback encoding

            # Start text processing in background
            executor.submit(process_document_async, text, filename, doc_id)

            # Return immediate response with document ID
            return jsonify({
                "document_id": doc_id,
                "file_info": {"filename": filename},
                "status": "processing",
                "message": "Text upload received. Processing started in background."
            })

    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        # Clean up saved file if it exists and an error occurred
        # Need to check if doc_id was generated and upload_path was set
        if 'doc_id' in locals() and doc_id in docs_store and 'upload_path' in docs_store[doc_id] and os.path.exists(docs_store[doc_id]['upload_path']):
             os.remove(docs_store[doc_id]['upload_path'])
             logger.info(f"Cleaned up temporary file {docs_store[doc_id]['upload_path']}")

        # If doc_id was generated, update its status to failed
        if 'doc_id' in locals() and doc_id in docs_store:
             docs_store[doc_id].update({
                  "status": "failed",
                  "error": str(e),
                  "processing_time": time.time() - docs_store[doc_id].get("processed_at", time.time()) # Calculate elapsed time
             })
        # If it failed before doc_id was even generated (less likely)
        else:
             logger.error("Upload failed before document ID could be generated.")
             return jsonify({'error': f"Upload failed: {str(e)}"}), 500


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

        # Use RAG pipeline to get answer
        answer_result = rag_pipeline.generate_answer(
            question=question,
            collection_id=doc_id, # Use doc_id as collection_id
            temperature=temperature,
            max_tokens=max_tokens
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
                         'similarity': source.get('score', 'N/A') # Use .get
                     })
                else:
                     logger.warning(f"Invalid chunk index {chunk_idx} found for document {doc_id}. Skipping source.")


        # Prepare response
        response = {
            'answer': answer_result.get('answer', 'Error retrieving answer.'),
            'sources': sources,
            'processing_time': answer_result.get('processing_time', 'N/A'),
            'error': answer_result.get('error') # Include error if present
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in answer endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# THIS IS THE FUNCTION THAT WAS PRIMARILY CHANGED TO FIX THE KEYERROR
@app.route('/api/document/<doc_id>/download', methods=['GET'])
def download_summary(doc_id):
    """Generate downloadable summary for a document"""
    # Use .get() to safely retrieve the document from docs_store
    doc = docs_store.get(doc_id)

    # Check if the document exists and has completed processing
    # Also check if 'chunks' key exists as it's needed for the summary
    if not doc or doc.get("status") != "completed" or 'chunks' not in doc:
        # Provide more specific error message based on status
        if not doc:
            return jsonify({'error': f'Document "{doc_id}" not found.'}), 404
        elif doc.get("status") == "processing":
            # Return 409 Conflict if still processing
            return jsonify({'error': f'Document "{doc_id}" is still processing. Please wait and try again.'}), 409
        elif doc.get("status") == "failed":
             return jsonify({'error': f'Document "{doc_id}" failed to process. Summary unavailable. Error: {doc.get("error", "Unknown error")}'}), 500
        else:
             return jsonify({'error': f'Document "{doc_id}" is not in a completed state or has no chunks.'}), 400


    # Ensure chunks have summaries before attempting to generate combined summary
    # (Based on your app.py code, summaries were removed from async processing temporarily,
    # you might need to re-add that or generate summaries on demand here if needed for the overview)
    # For now, let's fetch text content if summaries are not available
    all_relevant_text = [chunk.get('text') for chunk in doc.get('chunks', []) if chunk.get('text')] # Use .get for chunks list

    # Generate markdown summary
    md = [f"# Summary for {doc.get('filename', 'Document')}\n", # Use .get for safety
          f"- Document ID: {doc_id}",
          # Safely access 'processing_time' using .get() with a default value (0.0)
          f"- Processing Time: {doc.get('processing_time', 0.0):.2f} seconds",
          f"- Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
          "## Document Overview\n"]

    # Add document-level summary if there is text content to summarize
    doc_summary = "Combined summary unavailable." # Default message
    if all_relevant_text:
        try:
            # Generate combined summary from chunk texts if summaries are not stored
            # If you re-add summaries to chunks, you would use those here
            doc_summary = llm_manager.generate_combined_summary(all_relevant_text)
        except Exception as e:
            logger.error(f"Error generating combined summary for {doc_id}: {e}")
            doc_summary = f"Error generating combined summary: {str(e)}" # Report error in summary

    md.append(f"{doc_summary}\n\n## Chunk Summaries\n")

    # Add individual chunk summaries or excerpts
    for i, chunk in enumerate(doc.get('chunks', [])): # Use .get for chunks list
        # You might want to add summaries back to async processing,
        # or generate them on demand here if not stored.
        # For now, let's use excerpt or full text if summary is missing.
        chunk_content = chunk.get('summary') or chunk.get('excerpt') or chunk.get('text', 'Content unavailable.')
        md.append(f"### Chunk {i+1}\n{chunk_content}\n")


    markdown = '\n'.join(md)
    filename = f"{doc.get('filename', 'document')}_summary.md" # Use .get for safety

    # Return as JSON containing the markdown string and filename
    return jsonify({'markdown': markdown, 'filename': filename}), 200


if __name__ == '__main__':
    # Print startup information
    logger.info(f"Starting server with model: {DEFAULT_MODEL}")
    logger.info(f"Using device: {device}") # Use the locally determined device variable
    logger.info(f"Embeddings model: {EMBEDDINGS_MODEL}")

    # Use threaded=False for llama.cpp to avoid conflicts if necessary,
    # but ThreadPoolExecutor is used, so threading=True should be fine for Flask routes.
    # Use host and port from config.py
    # debug=DEBUG_MODE from config.py
    app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT, threaded=True)