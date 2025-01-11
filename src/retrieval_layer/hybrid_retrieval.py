import os
import yaml
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from flashrank import Ranker, RerankRequest
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetrieval:
    """Implements hybrid retrieval combining graph-based and embedding-based search."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the hybrid retrieval system."""
        self.config = self._load_config(config_path)
        self.ranker = self._initialize_ranker()
        self.index = None
        self.docstore = {}
        self.index_to_docstore_id = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _initialize_ranker(self) -> Ranker:
        """Initialize the reranking model."""
        try:
            cache_dir = self.config["ranking"]["cache_dir"]
            os.makedirs(cache_dir, exist_ok=True)
            return Ranker(
                model_name=self.config["ranking"]["model_name"],
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Error initializing ranker: {str(e)}")
            raise

    def build_index(self, documents: List[Document], batch_size: int = 25) -> None:
        """
        Build a FAISS index from document embeddings.
        
        Args:
            documents: List of documents with embeddings in metadata
            batch_size: Size of batches for processing
        """
        try:
            logger.info(f"Building index for {len(documents)} documents")
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Extract embeddings and create numpy array
                embeddings = []
                for doc in batch:
                    if 'embedding' not in doc.metadata:
                        logger.warning(f"Document missing embedding, skipping")
                        continue
                    embeddings.append(doc.metadata['embedding'])
                
                if not embeddings:
                    continue
                    
                batch_embeddings = np.array(embeddings, dtype=np.float32)
                
                # Initialize index with first batch
                if self.index is None:
                    self.index = faiss.IndexFlatIP(batch_embeddings.shape[1])
                
                # Normalize embeddings
                faiss.normalize_L2(batch_embeddings)
                
                # Add embeddings to index
                start_id = len(self.index_to_docstore_id)
                self.index.add(batch_embeddings)
                
                # Update docstore and mapping
                for j, doc in enumerate(batch):
                    if 'embedding' in doc.metadata:
                        doc_id = f"{start_id + j}"
                        self.docstore[doc_id] = doc
                        self.index_to_docstore_id[start_id + j] = doc_id
                
                logger.info(f"Processed batch {i//batch_size + 1}")
            
            logger.info(f"Index built with {len(self.index_to_docstore_id)} documents")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if self.index is None:
                raise ValueError("Index not built. Call build_index first.")
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            scores, indices = self.index.search(query_embedding, k)
            
            # Gather results
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:  # FAISS returns -1 for empty slots
                    continue
                doc_id = self.index_to_docstore_id.get(idx)
                if doc_id and doc_id in self.docstore:
                    results.append((self.docstore[doc_id], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def graph_search(
        self,
        graph_db,
        query: str,
        limit: int = 85
    ) -> List[Dict[str, Any]]:
        """
        Perform graph-based search.
        
        Args:
            graph_db: Graph database connection
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of graph search results
        """
        try:
            # Example graph query - modify based on your graph structure
            query = f"""
            MATCH p = (n)-[r]->(m)
            WHERE COUNT {{(n)--()}} > 30
            RETURN p AS Path
            LIMIT {limit}
            """
            
            response = graph_db.query(query)
            return response
            
        except Exception as e:
            logger.error(f"Error in graph search: {str(e)}")
            raise

    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on content and source.
        
        Args:
            results: List of search results
            rerank: Whether results are from reranking
            
        Returns:
            Deduplicated results list
        """
        try:
            seen = set()
            unique_results = []
            
            for result in results:
                if rerank:
                    identifier = (result['text'], str(result['meta']))
                else:
                    doc, score = result
                    identifier = (doc.page_content, str(doc.metadata.get('source', '')))
                
                if identifier not in seen:
                    seen.add(identifier)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in deduplication: {str(e)}")
            raise

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using the ranking model.
        
        Args:
            query: Search query
            results: Initial search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results list
        """
        try:
            # Prepare passages for reranking
            passages = []
            for idx, result in enumerate(results, start=1):
                if isinstance(result, tuple):  # Handle (doc, score) format
                    doc, score = result
                    passage = {
                        "id": idx,
                        "text": doc.page_content,
                        "meta": doc.metadata.get("source", "unknown"),
                        "score": float(score)
                    }
                else:  # Handle dictionary format
                    passage = {
                        "id": idx,
                        "text": result["text"],
                        "meta": result["meta"],
                        "score": result.get("score", 0.0)
                    }
                passages.append(passage)
            
            # Perform reranking
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.ranker.rerank(rerank_request)
            
            # Sort by score
            sorted_results = sorted(
                reranked_results,
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Apply top_k if specified
            if top_k:
                sorted_results = sorted_results[:top_k]
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            raise

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        graph_db,
        top_k: int = 100,
        rerank_top_k: Optional[int] = None,
        mode: str = "Hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining graph-based and embedding-based results.
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            graph_db: Graph database connection
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after reranking
            mode: Search mode ("Hybrid" or "Dense")
            
        Returns:
            Combined and ranked search results
        """
        try:
            results = []
            
            if mode == "Hybrid":
                # Perform graph search
                graph_results = self.graph_search(graph_db, query)
                results.extend(graph_results)
            
            # Perform embedding search
            embedding_results = self.similarity_search(query_embedding, k=top_k)
            results.extend(embedding_results)
            
            # Deduplicate results
            unique_results = self.deduplicate_results(results)
            
            # Rerank results
            reranked_results = self.rerank_results(
                query,
                unique_results,
                top_k=rerank_top_k
            )
            
            # Final deduplication
            final_results = self.deduplicate_results(reranked_results, rerank=True)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            traceback.print_exc()
            raise