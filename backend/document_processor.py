"""
document_processor.py - Handles document upload, processing, chunking, and caching for RAG system.

This class is responsible for:
- Saving uploaded documents
- Processing and chunking documents for vector storage
- Caching processed results for efficiency
- Supporting multiple file types (txt, pdf, docx, html, csv, json)
"""
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import logging
import hashlib
from pathlib import Path
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

# Langchain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, CSVLoader, JSONLoader
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document upload, processing, chunking, and caching for the RAG system.
    """
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        cache_dir: str = "doc_cache",
        max_workers: int = 4,
        skip_cleanup: bool = False
    ):
        """
        Initialize DocumentProcessor.
        Args:
            chunk_size: Target size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            cache_dir: Directory to cache processed documents
            max_workers: Number of parallel workers for processing
            skip_cleanup: Skip redundant document cleanup
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir
        self.skip_cleanup = skip_cleanup
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        os.makedirs(cache_dir, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ",", " ", ""],
            keep_separator=True
        )
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.doc': Docx2txtLoader,
            '.docx': Docx2txtLoader,
            '.html': BSHTMLLoader,
            '.htm': BSHTMLLoader,
            '.csv': CSVLoader,
            '.json': lambda path: JSONLoader(file_path=path, jq_schema='.', text_content=False)
        }

    def _get_loader(self, file_path: str) -> Any:
        """Return the appropriate loader for a file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in self.loaders:
            loader_class = self.loaders[file_extension]
            try:
                if callable(loader_class):
                    if isinstance(loader_class, type):
                        return loader_class(file_path)
                    else:
                        return loader_class(file_path)
                else:
                    raise ValueError(f"Loader for {file_extension} is not callable")
            except Exception as e:
                logger.error(f"Error creating loader for {file_path}: {str(e)}")
                raise
        elif os.path.isdir(file_path):
            return DirectoryLoader(file_path, glob="**/*.*")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _get_cache_key(self, file_path: str) -> str:
        """Generate a cache key for a file based on its path and stats."""
        try:
            file_stat = os.stat(file_path)
            unique_id = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(unique_id.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating cache key for {file_path}: {str(e)}")
            return hashlib.md5(file_path.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Check if a document is in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
                logger.info(f"Loaded {len(documents)} chunks from cache")
                return documents
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, documents: List[Document]) -> None:
        """Save processed documents to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(documents)} chunks to cache")
        except Exception as e:
            logger.warning(f"Failed to cache documents: {e}")

    def clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing."""
        if self.skip_cleanup:
            return text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()

    def smarter_split(self, texts: List[str]) -> List[Document]:
        """
        Split text intelligently to preserve context.
        1. Try to split on structural markers.
        2. If chunks are too large, fall back to recursive splitting.
        """
        documents = []
        def process_text(text, metadata=None):
            if metadata is None:
                metadata = {}
            clean = self.clean_text(text)
            structural_pattern = r'(?:^|\n)(?:#+\s|[A-Z][A-Z\s]+:|\d+\.\s|[IVXLCDM]+\.\s)'
            structural_chunks = re.split(structural_pattern, clean)
            if structural_chunks and all(len(chunk) <= self.chunk_size * 1.5 for chunk in structural_chunks if chunk):
                for i, chunk in enumerate(structural_chunks):
                    if not chunk.strip():
                        continue
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["chunk_type"] = "structural"
                    documents.append(Document(page_content=chunk.strip(), metadata=chunk_metadata))
            else:
                chunks = self.splitter.split_text(clean)
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    documents.append(Document(page_content=chunk.strip(), metadata=chunk_metadata))
        for text in texts:
            process_text(text)
        return documents

    def process_file(self, file_path: str, use_cache: bool = True) -> List[Document]:
        """
        Process a file: load, clean, split, and cache its content.
        Args:
            file_path: Path to the file
            use_cache: Whether to use cached results if available
        Returns:
            List of Document objects (chunks)
        """
        cache_key = self._get_cache_key(file_path)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached
        loader = self._get_loader(file_path)
        try:
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            documents = self.smarter_split(texts)
            self._save_to_cache(cache_key, documents)
            return documents
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}\n{traceback.format_exc()}")
            return []

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a raw text string into chunks.
        Args:
            text: The text to process
            metadata: Optional metadata to attach to each chunk
        Returns:
            List of Document objects (chunks)
        """
        try:
            return self.smarter_split([text])
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return []

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all files in a directory.
        Args:
            directory_path: Path to the directory
        Returns:
            List of Document objects (chunks)
        """
        loader = DirectoryLoader(directory_path, glob="**/*.*")
        try:
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            return self.smarter_split(texts)
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return []

    @staticmethod
    def save_upload(uploaded_file, directory: str = "uploads") -> str:
        """
        Save an uploaded file to the specified directory.
        Args:
            uploaded_file: File-like object
            directory: Directory to save the file
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        filename = os.path.basename(uploaded_file.filename)
        file_path = os.path.join(directory, filename)
        uploaded_file.save(file_path)
        return file_path

    def handle_document_input(self, uploaded_file: Any = None, text_input: str = "") -> List[Document]:
        """
        Handle either an uploaded file or direct text input.
        Args:
            uploaded_file: File-like object
            text_input: Raw text input
        Returns:
            List of Document objects (chunks)
        """
        if uploaded_file:
            file_path = self.save_upload(uploaded_file, self.cache_dir)
            return self.process_file(file_path)
        elif text_input:
            return self.process_text(text_input)
        else:
            logger.warning("No document input provided.")
            return []