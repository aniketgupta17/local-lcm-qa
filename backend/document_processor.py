# document_processor.py - Enhanced Document Processor
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

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor with better chunking and caching"""
    
    def __init__(
        self,
        chunk_size: int = 1200,        # Reduced chunk size for more precise retrieval
        chunk_overlap: int = 200,      # Increased overlap for better context preservation
        cache_dir: str = "doc_cache",
        max_workers: int = 4,          # Parallel processing
        skip_cleanup: bool = False     # Skip redundant document cleanup
    ):
        """
        Initialize with chunking parameters
        
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
        
        # Initialize text splitter with smarter separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ",", " ", ""],
            keep_separator=True
        )
        
        # Extended loader registry
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.doc': Docx2txtLoader,
            '.docx': Docx2txtLoader,
            '.html': BSHTMLLoader,  # Use better HTML loader
            '.htm': BSHTMLLoader,
            '.csv': CSVLoader,
            '.json': lambda path: JSONLoader(file_path=path, jq_schema='.', text_content=False)
        }
    
    def _get_loader(self, file_path: str) -> Any:
        """Get appropriate loader for file type with error handling"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in self.loaders:
            loader_class = self.loaders[file_extension]
            try:
                if callable(loader_class):
                    if isinstance(loader_class, type):  # It's a class
                        return loader_class(file_path)
                    else:  # It's a factory function
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
        """Generate a cache key for a file with file stats"""
        try:
            file_stat = os.stat(file_path)
            unique_id = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(unique_id.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating cache key for {file_path}: {str(e)}")
            # Fall back to just the file path if stat fails
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Check if document is in cache with error handling"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct Document objects
                documents = [
                    Document(
                        page_content=item['page_content'],
                        metadata=item['metadata']
                    )
                    for item in data
                ]
                logger.info(f"Loaded {len(documents)} chunks from cache")
                return documents
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, documents: List[Document]) -> None:
        """Save processed documents to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            # Convert Document objects to serializable dicts
            data = [
                {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in documents
            ]
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(documents)} chunks to cache")
        except Exception as e:
            logger.warning(f"Failed to cache documents: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing"""
        if self.skip_cleanup:  # Skip if requested
            return text
            
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove unprintable characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()
    
    def smarter_split(self, texts: List[str]) -> List[Document]:
        """
        Split text more intelligently to preserve context
        
        1. First attempt to split on structural markers
        2. If chunks are still too large, fall back to recursive splitting
        """
        documents = []
        
        # Process each text in parallel
        def process_text(text, metadata=None):
            if metadata is None:
                metadata = {}
                
            # Clean the text first
            clean = self.clean_text(text)
            
            # Try to split on structural boundaries first (headings, paragraphs)
            # This preserves semantic coherence better than character-based splits
            structural_pattern = r'(?:^|\n)(?:#+\s|[A-Z][A-Z\s]+:|\d+\.\s|[IVXLCDM]+\.\s)'
            structural_chunks = re.split(structural_pattern, clean)
            
            # If structural splitting created reasonable chunks, use those
            if structural_chunks and all(len(chunk) <= self.chunk_size * 1.5 for chunk in structural_chunks if chunk):
                # Process each structural chunk
                for i, chunk in enumerate(structural_chunks):
                    if not chunk.strip():
                        continue
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["chunk_type"] = "structural"
                    documents.append(Document(page_content=chunk.strip(), metadata=chunk_metadata))
            else:
                # Fall back to the recursive splitter for more precise control
                chunks = self.splitter.split_text(clean)
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["chunk_type"] = "recursive"
                    documents.append(Document(page_content=chunk.strip(), metadata=chunk_metadata))
        
        # Process texts in parallel
        futures = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            metadata = {"text_index": i, "source": "text_input"}
            future = self.executor.submit(process_text, text, metadata)
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result()
        
        return documents
    
    def process_file(self, file_path: str, use_cache: bool = True) -> List[Document]:
        """
        Load and chunk a document file with better error handling
        
        Args:
            file_path: Path to the document file
            use_cache: Whether to use cached results if available
            
        Returns:
            List of document chunks
        """
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(file_path)
            cached_docs = self._check_cache(cache_key)
            if cached_docs:
                logger.info(f"Using cached document ({len(cached_docs)} chunks)")
                return cached_docs
        
        # Process the file if not in cache
        try:
            logger.info(f"Loading file: {file_path}")
            loader = self._get_loader(file_path)
            raw_documents = loader.load()
            logger.info(f"Loaded {len(raw_documents)} document(s)")
            
            # Add source filename to metadata
            for doc in raw_documents:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["full_path"] = file_path
            
            # Split documents using smart splitting
            raw_texts = [doc.page_content for doc in raw_documents]
            metadata_list = [doc.metadata for doc in raw_documents]
            
            documents = []
            for text, metadata in zip(raw_texts, metadata_list):
                chunks = self.smarter_split([text])
                # Update metadata
                for chunk in chunks:
                    chunk.metadata.update(metadata)
                documents.extend(chunks)
            
            # Add chunk index to metadata
            for i, doc in enumerate(documents):
                doc.metadata["chunk_index"] = i
                doc.metadata["chunk_count"] = len(documents)
                doc.metadata["processing_time"] = time.time() - start_time
            
            # Cache the results if enabled
            if use_cache:
                cache_key = self._get_cache_key(file_path)
                self._save_to_cache(cache_key, documents)
            
            logger.info(f"Processed file into {len(documents)} chunks in {time.time() - start_time:.2f}s")
            return documents
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process raw text into chunks with better splitting
        
        Args:
            text: Raw text content
            metadata: Optional metadata to associate with text
            
        Returns:
            List of document chunks
        """
        start_time = time.time()
        
        if metadata is None:
            metadata = {"source": "text_input"}
        
        # Better handling of text chunks
        documents = self.smarter_split([text])
        
        # Update metadata
        for i, doc in enumerate(documents):
            doc.metadata.update(metadata)
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_count"] = len(documents)
            doc.metadata["processing_time"] = time.time() - start_time
        
        logger.info(f"Split text into {len(documents)} chunks in {time.time() - start_time:.2f}s")
        return documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all files in a directory with parallel processing
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of document chunks from all files
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        all_documents = []
        file_paths = []
        
        # Collect all valid file paths
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip hidden files and directories
                if file.startswith('.') or '/.' in file_path:
                    continue
                    
                # Skip unsupported file types silently
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension not in self.loaders.keys():
                    continue
                
                file_paths.append(file_path)
        
        # Process files in parallel
        futures = []
        for file_path in file_paths:
            future = self.executor.submit(self.process_file, file_path)
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                documents = future.result()
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Error processing file: {str(e)}")
        
        logger.info(f"Processed {len(all_documents)} chunks from directory {directory_path}")
        return all_documents
    
    @staticmethod
    def save_upload(uploaded_file, directory: str = "uploads") -> str:
        """
        Save an uploaded file to disk
        
        Args:
            uploaded_file: File object (with read and name attributes)
            directory: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        # Clean the filename to prevent path traversal
        filename = os.path.basename(uploaded_file.name)
        safe_filename = re.sub(r'[^\w\.-]', '_', filename)
        
        # Add timestamp to avoid overwrites
        file_path = os.path.join(directory, f"{int(time.time())}_{safe_filename}")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            
        return file_path
    
    def handle_document_input(
        self, 
        uploaded_file: Any = None, 
        text_input: str = ""
    ) -> List[Document]:
        """
        Process either uploaded file or direct text input
        
        Args:
            uploaded_file: File object (optional)
            text_input: Direct text input (optional)
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If neither file nor text is provided
        """
        if uploaded_file is not None:
            file_path = self.save_upload(uploaded_file)
            return self.process_file(file_path)
        elif text_input:
            return self.process_text(text_input)
        else:
            raise ValueError("Either a file or text input must be provided")