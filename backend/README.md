# Scientific Document Q&A Backend

This backend powers the Scientific Document Q&A System, enabling local, privacy-preserving document analysis and question answering using open-source Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview
- **Framework:** Python + Flask
- **Core Features:**
  - Upload and process documents (PDF, DOCX, TXT, HTML, CSV, JSON)
  - Chunking, summarization, and vector storage (ChromaDB)
  - Hybrid semantic/keyword search for retrieval
  - LLM-powered Q&A and summarization (HuggingFace, transformers, llama.cpp)
  - All processing is local—your data never leaves your machine

---

## 📁 Folder Structure

```
backend/
├── app.py                # Main Flask app and API endpoints
├── config.py             # Centralized configuration (models, paths, settings)
├── document_processor.py # Handles file upload, chunking, and caching
├── llm_manager.py        # Manages LLM inference (summarization, Q&A)
├── rag_pipeline.py       # Orchestrates RAG workflow (retrieval, reranking, answer)
├── vector_store.py       # Vector DB (ChromaDB) and embedding management
├── utils.py              # Utility functions (logging, ID generation, etc.)
├── models/               # (Optional) Local model files
├── uploads/              # Uploaded documents
├── vector_db/            # ChromaDB persistent storage
├── cache/                # Chunk and embedding cache
├── logs/                 # Log files
```

---

## ⚙️ Configuration
- All settings (model, device, chunk size, etc.) are in `config.py` and can be overridden by environment variables.
- Default LLM: `mistralai/Mistral-7B-Instruct-v0.3` (HuggingFace)
- Default embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Device: CPU, CUDA, or MPS (auto-detect)

---

## 🛠️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # (You may need additional system packages for some document types)
   ```

2. **(Optional) Set environment variables:**
   - `DEFAULT_MODEL`, `EMBEDDINGS_MODEL`, `DEVICE`, etc.

3. **Start the backend:**
   ```bash
   python app.py
   # By default runs on http://localhost:8000
   ```

---

## 🔗 API Endpoints

- `GET  /api/health` — Health check and system info
- `GET  /api/documents` — List all processed documents
- `POST /api/upload` — Upload a document (file or text)
- `GET  /api/documents/<doc_id>` — Get document details
- `DELETE /api/documents/<doc_id>` — Delete a document
- `POST /api/answer` — Ask a question about a document
- `GET  /api/document/<doc_id>/download` — Download chunk summaries

### Example: Upload a Document (cURL)
```bash
curl -F "file=@yourfile.pdf" http://localhost:8001/api/upload
```

### Example: Ask a Question
```bash
curl -X POST http://localhost:8001/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the summary?", "document_id": "<doc_id>"}'
```


---

## 🐞 Troubleshooting
- Check `logs/` for error logs
- Ensure all dependencies are installed (see `requirements.txt`)
- For GPU, set `DEVICE=cuda` and ensure CUDA drivers are installed
- For HuggingFace API, set `HF_API_TOKEN` in your ---

## 🧩 Main Components
- **DocumentProcessor:** Loads, chunks, and caches documents
- **VectorStore:** Stores and retrieves embeddings (ChromaDB)
- **RAGPipeline:** Retrieval, reranking, and answer generation
- **LLMManager:** Manages LLM inference (summarization, Q&A)

## 📚 References
- [Flask](https://flask.palletsprojects.com/)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [LangChain](https://python.langchain.com/)

---

**All processing is local and private. Your data never leaves your machine.** 