"""
vector_store.py - Handles vector storage, embedding management, and collection retrieval for the RAG system using ChromaDB.

This class is responsible for:
- Creating and managing vector stores (ChromaDB)
- Storing and retrieving embeddings and document collections
- Supporting analytics for uploaded documents
- Using local sentence-transformers for embeddings
"""
import os
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles vector storage, embedding management, and collection retrieval for the RAG system using ChromaDB.
    """
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "default_collection"
    ):
        """
        Initialize the VectorStore with ChromaDB and a local embedding model.
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model_name: Name of the local sentence-transformers model
            collection_name: Default collection name
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        logger.info(f"ChromaDB initialized at {self.persist_directory} with collection '{self.collection_name}' and embedding model '{self.embedding_model_name}'")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using the local model."""
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def add_documents(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to ChromaDB with embeddings.
        Args:
            doc_id: Unique document identifier
            chunks: List of dicts with 'text' and 'metadata'
        """
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        # Add doc_id to each metadata for easy filtering
        for m in metadatas:
            m['doc_id'] = doc_id
        embeddings = self.embed_texts(texts)
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"Added {len(chunks)} chunks for document {doc_id} to ChromaDB.")

    def get_document_analytics(self, doc_id: str) -> Dict[str, Any]:
        """
        Return analytics for a document: chunk count, chunk metadata, etc.
        Args:
            doc_id: Unique document identifier
        Returns:
            Dict with analytics
        """
        # Query all chunks for this doc_id
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas", "documents"]
        )
        chunk_count = len(results["ids"])
        chunk_metadatas = results["metadatas"]
        return {
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "chunk_metadatas": chunk_metadatas
        }

    def list_collections(self) -> List[str]:
        """List all collection names in ChromaDB."""
        return [c.name for c in self.client.list_collections()]

    def reset(self):
        """Delete all collections (for testing/debug)."""
        for c in self.client.list_collections():
            self.client.delete_collection(c.name)
        logger.info("All ChromaDB collections deleted.")

    def search(self, collection_id: str, query: str, top_k: int = 5, hybrid_alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search: combine semantic (vector) and keyword (explicit memory) retrieval.
        Args:
            collection_id: Document collection ID (doc_id)
            query: User query string
            top_k: Number of results to return
            hybrid_alpha: Weight for semantic (1.0) vs keyword (0.0) search
        Returns:
            List of dicts: {text, id, score, semantic_score, keyword_score, metadata}
        """
        # 1. Semantic search (vector similarity)
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"doc_id": collection_id},
            include=["documents", "metadatas", "distances"]
        )
        semantic_chunks = {}
        for i, (doc, meta, dist) in enumerate(zip(
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
            semantic_results["distances"][0]
        )):
            semantic_score = 1.0 - dist if dist is not None else 0.0
            idx = meta.get("chunk_index", i)
            semantic_chunks[idx] = {
                "text": doc,
                "id": idx,
                "score": semantic_score,
                "semantic_score": semantic_score,
                "keyword_score": 0.0,
                "metadata": meta
            }
        # 2. Keyword search (explicit memory): fetch all chunks for doc_id, filter by query in text
        keyword_results = self.collection.get(
            where={"doc_id": collection_id},
            include=["documents", "metadatas"]
        )
        for i, (doc, meta) in enumerate(zip(
            keyword_results["documents"],
            keyword_results["metadatas"]
        )):
            if query.lower() in doc.lower():
                idx = meta.get("chunk_index", i)
                if idx in semantic_chunks:
                    semantic_chunks[idx]["keyword_score"] = 1.0
                else:
                    semantic_chunks[idx] = {
                        "text": doc,
                        "id": idx,
                        "score": 0.0,
                        "semantic_score": 0.0,
                        "keyword_score": 1.0,
                        "metadata": meta
                    }
        # 3. Combine scores (hybrid)
        results = []
        for idx, chunk in semantic_chunks.items():
            hybrid_score = hybrid_alpha * chunk["semantic_score"] + (1 - hybrid_alpha) * chunk["keyword_score"]
            chunk["score"] = hybrid_score
            results.append(chunk)
        # 4. Sort by hybrid score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]