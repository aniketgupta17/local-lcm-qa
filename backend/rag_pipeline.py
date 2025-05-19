"""
rag_pipeline.py - Orchestrates the Retrieval-Augmented Generation (RAG) process for the system.

This class is responsible for:
- Managing the RAG workflow (retrieval, reranking, answer generation)
- Using vector store, LLM manager, and prompt templates
- Supporting hybrid search and reranking
- Providing both sync and async answer generation
"""
import time
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) process for the system.
    Handles retrieval, reranking, and answer generation using vector store and LLM manager.
    """
    
    def __init__(
        self,
        vector_store,
        llm_manager,
        rerank_top_k: int = 10,
        final_top_k: int = 3,
        use_hybrid_search: bool = True,
        cache_size: int = 32,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 4
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store instance for document retrieval
            llm_manager: LLM manager for generating answers
            rerank_top_k: Number of documents to retrieve before reranking
            final_top_k: Number of documents to use after reranking
            use_hybrid_search: Whether to use hybrid search
            cache_size: Size of the LRU cache for query results
            executor: Optional ThreadPoolExecutor for parallel processing
            max_workers: Number of workers for the executor if not provided
        """
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = 0.7  # Weight for semantic search vs keyword search
        
        # Resource management
        self.executor = executor if executor else ThreadPoolExecutor(max_workers=max_workers)
        self.process_lock = Lock()
        
        # Caching
        self.query_cache = {}
        self.cache_lock = Lock()
        
        # Stats
        self.request_count = 0
        self.cache_hits = 0
        
        # Prompt templates
        self.qa_prompt_template = """
You are a scientific assistant. Use ONLY the provided context (including both the summary and content of each chunk) to answer the question. 
If the answer is not in the context, say "Not found in the provided documents."
Cite the [Document #] in your answer where relevant.

Context:
{context}

Question: {question}

Answer (with citations):
"""

        self.reranking_prompt_template = """Please score the relevance of this passage to the given question on a scale of 1-10, where 10 is perfectly relevant and 1 is completely irrelevant.

Question: {question}

Passage: {passage}

Relevance Score (1-10):"""

    @lru_cache(maxsize=32)
    def _cached_retrieve_documents(
        self, collection_id: str, query: str, top_k: int, hybrid_alpha: float
    ) -> List[Dict[str, Any]]:
        """Cache-enabled document retrieval"""
        return self.vector_store.search(collection_id, query, top_k, hybrid_alpha)
    
    async def _aio_retrieve_documents(
        self, collection_id: str, query: str, top_k: int, hybrid_alpha: float
    ) -> List[Dict[str, Any]]:
        """Async document retrieval using ThreadPoolExecutor"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._cached_retrieve_documents(collection_id, query, top_k, hybrid_alpha)
        )
    
    async def _aio_rerank_documents(
        self, documents: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Async document reranking using LLM"""
        if not documents:
            return []
        
        async def _score_document(doc):
            prompt = self.reranking_prompt_template.format(
                question=query,
                passage=doc["text"]
            )
            
            # Use the executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.llm_manager.generate(prompt, temperature=0.1, max_tokens=10)
            )
            
            try:
                # Extract the score from the LLM's output
                score_text = result.strip()
                # Try to extract just the number
                import re
                match = re.search(r"(\d+)", score_text)
                if match:
                    score = float(match.group(1))
                else:
                    score = float(score_text) if score_text.isdigit() else 5.0
                
                # Normalize to 0-1 range
                score = min(max(score / 10.0, 0.0), 1.0)
            except Exception as e:
                logger.warning(f"Error parsing reranking score: {e}")
                score = 0.5  # Default to middle score on error
            
            return {**doc, "rerank_score": score}
        
        # Process documents in parallel
        tasks = [_score_document(doc) for doc in documents]
        reranked_docs = await asyncio.gather(*tasks)
        
        # Sort by reranking score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepare context from retrieved documents, including both summaries and chunk texts."""
        if not documents:
            return "", []
        context_parts = []
        sources = []
        for i, doc in enumerate(documents):
            summary = doc["metadata"].get("summary", "[No summary available]")
            context_parts.append(f"[Document {i+1}]\nSummary: {summary}\nContent: {doc['text']}\n")
            sources.append({
                "chunk_index": doc["id"],
                "score": doc.get("rerank_score", doc.get("score", 0))
            })
        return "\n".join(context_parts), sources
    
    def generate_answer(
        self,
        question: str,
        collection_id: str,
        temperature: float = 0.1,
        max_tokens: int = 300,
        use_reranking: bool = True,
        hybrid_alpha: float = None  # Added parameter with default None
    ) -> Dict[str, Any]:
        """
        Generate an answer for a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            collection_id: ID of the document collection to search
            temperature: LLM temperature parameter
            max_tokens: Maximum number of tokens to generate
            use_reranking: Whether to use reranking for better results
            hybrid_alpha: Weight balance between semantic (1.0) and keyword search (0.0).
                         If None, uses the class default value.
            
        Returns:
            Dictionary with answer, sources, and processing time
        """
        start_time = time.time()
        
        # Use provided hybrid_alpha or fall back to class default
        if hybrid_alpha is None:
            hybrid_alpha = self.hybrid_alpha
        
        # Check cache first
        cache_key = f"{collection_id}:{question}:{temperature}:{max_tokens}:{hybrid_alpha}"
        with self.cache_lock:
            if cache_key in self.query_cache:
                self.cache_hits += 1
                logger.info(f"Cache hit for query: {question}")
                return self.query_cache[cache_key]
        
        try:
            # Step 1: Initial retrieval
            retrieved_docs = self._cached_retrieve_documents(
                collection_id, 
                question, 
                self.rerank_top_k if use_reranking else self.final_top_k,
                hybrid_alpha
            )
            
            # Step 2: Reranking (if enabled)
            if use_reranking and retrieved_docs:
                try:
                    # Use asyncio for parallel reranking
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    retrieved_docs = loop.run_until_complete(
                        self._aio_rerank_documents(retrieved_docs, question)
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Error during reranking: {e}")
            
            # Step 3: Select top documents after reranking
            final_docs = retrieved_docs[:self.final_top_k]
            
            # Step 4: Prepare context from selected documents
            context, sources = self._prepare_context(final_docs)
            
            # Step 5: Generate answer using LLM
            if context:
                prompt = self.qa_prompt_template.format(
                    context=context,
                    question=question
                )
                
                answer = self.llm_manager.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                answer = "I don't have enough information to answer this question based on the document content."
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }
            
            # Cache the result
            with self.cache_lock:
                self.query_cache[cache_key] = result
                # Simple cache size management
                if len(self.query_cache) > 100:
                    # Remove oldest entries
                    keys_to_remove = list(self.query_cache.keys())[:-50]
                    for key in keys_to_remove:
                        del self.query_cache[key]
            
            logger.info(f"Generated answer in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "Sorry, I encountered an error while trying to answer your question.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def generate_combined_summary(
        self, documents: List[str], max_tokens: int = 500
    ) -> str:
        """
        Generate a combined summary from multiple document chunks.
        
        Args:
            documents: List of document texts to summarize
            max_tokens: Maximum number of tokens for the summary
            
        Returns:
            Generated summary
        """
        # Combine documents with markers
        combined_text = "\n\n---\n\n".join(documents)
        
        # Generate summary
        summary_prompt = f"""Please provide a concise summary of the following document sections:

{combined_text}

Summary:"""
        
        try:
            return self.llm_manager.generate(
                summary_prompt, 
                temperature=0.1,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Error generating combined summary: {e}")
            return "Error generating summary."