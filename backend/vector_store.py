# vector_store.py - Vector store and embedding manager
import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import faiss
import logging
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import threading
import uuid
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Enhanced vector store with FAISS integration and hybrid search capability
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_db_path: str = "./vector_db",
        device: Optional[torch.device] = None,
        use_faiss: bool = True,
        dim_reduction: Optional[int] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_hybrid_search: bool = True
    ):
        """
        Initialize the vector store
        
        Args:
            model_name: Name of the embedding model
            vector_db_path: Path to store vector databases
            device: PyTorch device (cuda, cpu, mps)
            use_faiss: Whether to use FAISS for faster similarity search
            dim_reduction: Dimensionality reduction target (None to disable)
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings
            use_hybrid_search: Whether to use hybrid search (semantic + keyword)
        """
        self.model_name = model_name
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True, parents=True)
        self.use_faiss = use_faiss
        self.dim_reduction = dim_reduction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_hybrid_search = use_hybrid_search
        
        # Set device
        self.device = device if device is not None else self._get_default_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model with optimization
        self.model_lock = threading.RLock()
        self._initialize_model()
        
        # Initialize PCA if dimension reduction requested
        self.pca = None
        if self.dim_reduction is not None:
            logger.info(f"Will reduce embeddings to {self.dim_reduction} dimensions")
        
        # Collections storage
        self.collections = {}
        self.tfidf_vectorizers = {}
        self.load_collections()
    
    def _get_default_device(self) -> torch.device:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model with optimizations"""
        logger.info(f"Loading embedding model: {self.model_name}")
        with self.model_lock:
            try:
                # Optimize loading
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                # Apply half-precision if using GPU to save memory
                if self.device.type == "cuda":
                    self.model.half()  # Convert to FP16 for faster inference on GPU
                # Set proper batch size for inference
                self.model.max_seq_length = 512  # Cap sequence length for speed
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def load_collections(self) -> None:
        """Load existing collections from disk"""
        try:
            collections_path = self.vector_db_path / "collections.json"
            if collections_path.exists():
                with open(collections_path, "r") as f:
                    collection_metadata = json.load(f)
                
                for collection_id, metadata in collection_metadata.items():
                    self._load_collection(collection_id, metadata)
                
                logger.info(f"Loaded {len(collection_metadata)} collections")
            else:
                logger.info("No existing collections found")
        except Exception as e:
            logger.warning(f"Error loading collections: {str(e)}")
    
    def _load_collection(self, collection_id: str, metadata: Dict[str, Any]) -> None:
        """Load a specific collection from disk"""
        try:
            # Load FAISS index if available
            index_path = self.vector_db_path / f"{collection_id}.faiss"
            if self.use_faiss and index_path.exists():
                index = faiss.read_index(str(index_path))
            else:
                # Fall back to numpy
                vectors_path = self.vector_db_path / f"{collection_id}.npy"
                if vectors_path.exists():
                    vectors = np.load(str(vectors_path))
                    if self.use_faiss:
                        # Create a new FAISS index
                        dim = vectors.shape[1]
                        index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
                        if vectors.shape[0] > 0:  # Only add if there are vectors
                            if self.normalize_embeddings:
                                faiss.normalize_L2(vectors)
                            index.add(vectors)
                    else:
                        index = vectors
                else:
                    # Create empty index
                    dim = metadata.get("dim", 384)
                    if self.use_faiss:
                        index = faiss.IndexFlatIP(dim)
                    else:
                        index = np.zeros((0, dim), dtype=np.float32)
            
            # Load texts
            texts_path = self.vector_db_path / f"{collection_id}_texts.pkl"
            if texts_path.exists():
                with open(texts_path, "rb") as f:
                    texts = pickle.load(f)
            else:
                texts = []
            
            # Initialize TF-IDF if using hybrid search
            if self.use_hybrid_search:
                tfidf_path = self.vector_db_path / f"{collection_id}_tfidf.pkl"
                if tfidf_path.exists():
                    with open(tfidf_path, "rb") as f:
                        self.tfidf_vectorizers[collection_id] = pickle.load(f)
                else:
                    self.tfidf_vectorizers[collection_id] = TfidfVectorizer(
                        lowercase=True, stop_words="english", ngram_range=(1, 2)
                    )
                    if texts:
                        self.tfidf_vectorizers[collection_id].fit(texts)
            
            # Store in memory
            self.collections[collection_id] = {
                "index": index,
                "texts": texts,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error loading collection {collection_id}: {str(e)}")
    
    def _save_collection(self, collection_id: str) -> None:
        """Save a collection to disk"""
        try:
            collection = self.collections[collection_id]
            
            # Save index/vectors
            if self.use_faiss:
                index_path = self.vector_db_path / f"{collection_id}.faiss"
                faiss.write_index(collection["index"], str(index_path))
            else:
                vectors_path = self.vector_db_path / f"{collection_id}.npy"
                np.save(str(vectors_path), collection["index"])
            
            # Save texts
            texts_path = self.vector_db_path / f"{collection_id}_texts.pkl"
            with open(texts_path, "wb") as f:
                pickle.dump(collection["texts"], f)
            
            # Save TF-IDF vectorizer
            if self.use_hybrid_search and collection_id in self.tfidf_vectorizers:
                tfidf_path = self.vector_db_path / f"{collection_id}_tfidf.pkl"
                with open(tfidf_path, "wb") as f:
                    pickle.dump(self.tfidf_vectorizers[collection_id], f)
            
            # Update collections metadata
            self._save_collections_metadata()
            
        except Exception as e:
            logger.error(f"Error saving collection {collection_id}: {str(e)}")
    
    def _save_collections_metadata(self) -> None:
        """Save collections metadata to disk"""
        try:
            collections_metadata = {
                collection_id: collection["metadata"] 
                for collection_id, collection in self.collections.items()
            }
            
            collections_path = self.vector_db_path / "collections.json"
            with open(collections_path, "w") as f:
                json.dump(collections_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving collections metadata: {str(e)}")
    
    def create_collection(self, collection_id: str, dim: int = 384) -> None:
        """Create a new empty collection"""
        if collection_id in self.collections:
            logger.warning(f"Collection {collection_id} already exists. Skipping creation.")
            return
        
        # Create empty index
        if self.use_faiss:
            index = faiss.IndexFlatIP(dim)
        else:
            index = np.zeros((0, dim), dtype=np.float32)
        
        # Create collection
        self.collections[collection_id] = {
            "index": index,
            "texts": [],
            "metadata": {
                "dim": dim,
                "count": 0,
                "created_at": time.time()
            }
        }
        
        # Initialize TF-IDF vectorizer
        if self.use_hybrid_search:
            self.tfidf_vectorizers[collection_id] = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2)
            )
        
        # Save to disk
        self._save_collection(collection_id)
        logger.info(f"Created new collection: {collection_id}")
    
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        if collection_id not in self.collections:
            logger.warning(f"Collection {collection_id} does not exist. Nothing to delete.")
            return False
        
        try:
            # Remove from memory
            del self.collections[collection_id]
            if collection_id in self.tfidf_vectorizers:
                del self.tfidf_vectorizers[collection_id]
            
            # Remove files
            index_path = self.vector_db_path / f"{collection_id}.faiss"
            vectors_path = self.vector_db_path / f"{collection_id}.npy"
            texts_path = self.vector_db_path / f"{collection_id}_texts.pkl"
            tfidf_path = self.vector_db_path / f"{collection_id}_tfidf.pkl"
            
            for path in [index_path, vectors_path, texts_path, tfidf_path]:
                if path.exists():
                    path.unlink()
            
            # Update metadata
            self._save_collections_metadata()
            
            logger.info(f"Deleted collection: {collection_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_id}: {str(e)}")
            return False
    
    def _fit_pca_reducer(self, embeddings: np.ndarray) -> None:
        """Initialize and fit PCA for dimension reduction"""
        if embeddings.shape[0] < 2:
            logger.warning("Not enough embeddings to fit PCA. Skipping dimension reduction.")
            return
        
        try:
            self.pca = PCA(n_components=self.dim_reduction)
            self.pca.fit(embeddings)
            logger.info(f"PCA fitted with {self.pca.n_components_} components")
            
            # Save PCA model
            pca_path = self.vector_db_path / "pca.pkl"
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)
        except Exception as e:
            logger.error(f"Error fitting PCA: {str(e)}")
            self.pca = None
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions using PCA"""
        if self.dim_reduction is None or self.pca is None:
            return embeddings
        
        try:
            reduced = self.pca.transform(embeddings)
            logger.debug(f"Reduced dimensions from {embeddings.shape[1]} to {reduced.shape[1]}")
            return reduced
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            return embeddings
    
    def _try_load_pca(self) -> None:
        """Try to load saved PCA model"""
        pca_path = self.vector_db_path / "pca.pkl"
        if pca_path.exists():
            try:
                with open(pca_path, "rb") as f:
                    self.pca = pickle.load(f)
                logger.info(f"Loaded PCA with {self.pca.n_components_} components")
            except Exception as e:
                logger.warning(f"Error loading PCA: {str(e)}")
                self.pca = None
    
    @lru_cache(maxsize=100)
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text with caching"""
        with self.model_lock:
            return self.model.encode(text, show_progress_bar=False)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        # Process in batches
        all_embeddings = []
        with self.model_lock:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=(len(texts) > 100),
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
        
        # Combine results
        embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Apply dimension reduction if configured
        if self.dim_reduction is not None:
            # Try to load existing PCA first
            if self.pca is None:
                self._try_load_pca()
            
            # Fit PCA if needed
            if self.pca is None and len(embeddings) >= 10:
                self._fit_pca_reducer(embeddings)
            
            # Apply dimension reduction
            if self.pca is not None:
                embeddings = self._reduce_dimensions(embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s")
        return embeddings
    
    def add_documents(
        self, collection_id: str, texts: List[str], embeddings: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add documents to a collection
        
        Args:
            collection_id: ID of the collection to add to
            texts: List of text strings
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Success flag
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return False
        
        # Create collection if it doesn't exist
        if collection_id not in self.collections:
            dim = embeddings.shape[1] if embeddings is not None else 384
            self.create_collection(collection_id, dim=dim)
        
        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self.embed_documents(texts)
        
        try:
            collection = self.collections[collection_id]
            
            # Add to index
            if self.use_faiss:
                if self.normalize_embeddings:
                    faiss.normalize_L2(embeddings.astype('float32'))
                collection["index"].add(embeddings.astype('float32'))
            else:
                # Using numpy
                if len(collection["texts"]) == 0:
                    collection["index"] = embeddings
                else:
                    collection["index"] = np.vstack([collection["index"], embeddings])
            
            # Update texts
            collection["texts"].extend(texts)
            
            # Update TF-IDF vectorizer
            if self.use_hybrid_search:
                self.tfidf_vectorizers[collection_id].fit(collection["texts"])
            
            # Update metadata
            collection["metadata"]["count"] = len(collection["texts"])
            collection["metadata"]["last_updated"] = time.time()
            
            # Save to disk
            self._save_collection(collection_id)
            
            logger.info(f"Added {len(texts)} documents to collection {collection_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {collection_id}: {str(e)}")
            return False
    
    def _semantic_search(
        self, collection_id: str, query_embedding: np.ndarray, top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        if collection_id not in self.collections:
            logger.warning(f"Collection {collection_id} does not exist")
            return []
        
        collection = self.collections[collection_id]
        
        if self.use_faiss:
            # Normalize query vector for inner product
            if self.normalize_embeddings:
                faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Search FAISS index
            scores, indices = collection["index"].search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(top_k, len(collection["texts"]))
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(collection["texts"]):
                    continue  # Skip invalid indices
                results.append({
                    "id": idx,
                    "score": float(score),
                    "text": collection["texts"][idx]
                })
        else:
            # Numpy-based search
            if len(collection["texts"]) == 0:
                return []
            
            # Normalize embeddings for dot product similarity
            if self.normalize_embeddings:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Calculate similarities
            similarities = np.dot(collection["index"], query_embedding)
            
            # Get top k
            if len(similarities) <= top_k:
                indices = np.argsort(similarities)[::-1]
                scores = similarities[indices]
            else:
                indices = np.argpartition(similarities, -top_k)[-top_k:]
                indices = indices[np.argsort(similarities[indices])[::-1]]
                scores = similarities[indices]
            
            results = []
            for i, (idx, score) in enumerate(zip(indices, scores)):
                results.append({
                    "id": int(idx),
                    "score": float(score),
                    "text": collection["texts"][idx]
                })
        
        return results
    
    def _keyword_search(
        self, collection_id: str, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using TF-IDF similarity"""
        if not self.use_hybrid_search or collection_id not in self.tfidf_vectorizers:
            return []
        
        try:
            collection = self.collections[collection_id]
            vectorizer = self.tfidf_vectorizers[collection_id]
            
            # Transform query and documents
            query_vec = vectorizer.transform([query])
            doc_vectors = vectorizer.transform(collection["texts"])
            
            # Calculate similarities
            similarities = (query_vec @ doc_vectors.T).toarray()[0]
            
            # Get top k
            if len(similarities) <= top_k:
                indices = np.argsort(similarities)[::-1]
                scores = similarities[indices]
            else:
                indices = np.argpartition(similarities, -top_k)[-top_k:]
                indices = indices[np.argsort(similarities[indices])[::-1]]
                scores = similarities[indices]
            
            results = []
            for i, (idx, score) in enumerate(zip(indices, scores)):
                if score > 0:  # Only include matches
                    results.append({
                        "id": int(idx),
                        "score": float(score),
                        "text": collection["texts"][idx]
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def search(
        self, collection_id: str, query: str, top_k: int = 5, hybrid_alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            collection_id: ID of the collection to search
            query: Search query text
            top_k: Number of results to return
            hybrid_alpha: Weight for semantic search (1-hybrid_alpha for keyword search)
            
        Returns:
            List of search results with scores
        """
        start_time = time.time()
        
        if collection_id not in self.collections:
            logger.warning(f"Collection {collection_id} does not exist")
            return []
        
        # Generate query embedding
        query_embedding = self._embed_text(query)
        
        # Apply dimension reduction if configured
        if self.dim_reduction is not None and self.pca is not None:
            query_embedding = self._reduce_dimensions(query_embedding.reshape(1, -1))[0]
        
        # Perform semantic search
        semantic_results = self._semantic_search(collection_id, query_embedding, top_k)
        
        # Perform hybrid search if enabled
        if self.use_hybrid_search and hybrid_alpha < 1.0:
            # Get more candidates for better hybrid results
            extended_k = min(top_k * 3, len(self.collections[collection_id]["texts"]))
            keyword_results = self._keyword_search(collection_id, query, extended_k)
            
            if keyword_results:
                # Combine results
                combined_results = {}
                
                # Add semantic results with weight
                for result in semantic_results:
                    combined_results[result["id"]] = {
                        "id": result["id"],
                        "score": result["score"] * hybrid_alpha,
                        "text": result["text"],
                        "semantic_score": result["score"]
                    }
                
                # Add keyword results with weight
                for result in keyword_results:
                    if result["id"] in combined_results:
                        combined_results[result["id"]]["score"] += result["score"] * (1 - hybrid_alpha)
                        combined_results[result["id"]]["keyword_score"] = result["score"]
                    else:
                        combined_results[result["id"]] = {
                            "id": result["id"],
                            "score": result["score"] * (1 - hybrid_alpha),
                            "text": result["text"],
                            "keyword_score": result["score"]
                        }
                
                # Sort by combined score
                results = list(combined_results.values())
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:top_k]
            else:
                results = semantic_results
        else:
            results = semantic_results
        
        logger.info(f"Search completed in {time.time() - start_time:.4f}s")
        return results[:top_k]