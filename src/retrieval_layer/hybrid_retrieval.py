import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from flashrank import Ranker, RerankRequest
from ..processing_layer.embedding_generator import EmbeddingGenerator
from ..processing_layer.graph_constructor import GraphConstructor
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
        self.embedding_generator = EmbeddingGenerator(config_path)
        
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

    def build_index(self, documents: List[Document]) -> None:
        """
        Add documents to the Chroma vector store.
        
        Args:
            documents: List of documents with embeddings in metadata
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.embedding_generator.save_embeddings(documents)
            logger.info("Documents successfully added to vector store")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using Chroma vector store.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Use Chroma's similarity search
            results = self.embedding_generator.vector_store.similarity_search_with_score(
                query_embedding,
                k=k
            )
            
            return [(doc, float(score)) for doc, score in results]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def graph_search(
        self,
        graph: GraphConstructor,
        query: str,
        limit: int = 85
    ) -> List[Dict[str, Any]]:
        """
        Perform graph-based search using NetworkX.
        
        Args:
            graph: GraphConstructor instance
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of graph search results
        """
        try:
            # Use NetworkX to find important nodes and their relationships
            results = []
            
            # Query the graph using NetworkX's centrality measures
            query_params = {
                'type': 'subgraph',
                'params': {
                    'nodes': list(graph.graph.nodes())[:limit]  # Get top nodes by degree
                }
            }
            
            # Get subgraph of important nodes
            subgraph_data = graph.query_graph(query_params)
            
            # Convert NetworkX results to a format compatible with our system
            for result in subgraph_data:
                for node, node_data in result['nodes']:
                    # Include node and its immediate neighbors
                    neighbors = list(graph.graph.neighbors(node))
                    edges = []
                    for neighbor in neighbors:
                        edge_data = graph.graph.get_edge_data(node, neighbor)
                        edges.append({
                            'source': node,
                            'target': neighbor,
                            'type': edge_data.get('type', 'unknown'),
                            'properties': edge_data
                        })
                    
                    results.append({
                        'node': {
                            'id': node,
                            'type': node_data.get('type', 'unknown'),
                            'properties': node_data
                        },
                        'edges': edges
                    })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
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
                if isinstance(result, tuple):
                    # Handle (doc, score) format from similarity search
                    doc, score = result
                    identifier = (doc.page_content, str(doc.metadata.get('source', '')))
                else:
                    # Handle dictionary format from reranking or graph search
                    text = result.get('text', '')
                    if not text and 'node' in result:
                        # Handle graph search results
                        text = str(result['node'].get('properties', {}))
                    meta = result.get('meta', '')
                    if not meta and 'node' in result:
                        meta = result['node'].get('id', '')
                    identifier = (text, str(meta))
                
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
                elif 'node' in result:  # Handle graph search results
                    node = result['node']
                    passage = {
                        "id": idx,
                        "text": str(node.get('properties', {})),
                        "meta": node.get('id', 'unknown'),
                        "score": len(result.get('edges', []))  # Use number of edges as initial score
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
        graph: Optional[GraphConstructor] = None,
        top_k: int = 100,
        rerank_top_k: Optional[int] = None,
        mode: str = "Hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining graph-based and embedding-based results.
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            graph: GraphConstructor instance
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after reranking
            mode: Search mode ("Hybrid" or "Dense")
            
        Returns:
            Combined and ranked search results
        """
        try:
            results = []
            
            if mode == "Hybrid" and graph is not None:
                try:
                    # Perform graph search if available
                    graph_results = self.graph_search(graph, query)
                    results.extend(graph_results)
                except Exception as e:
                    logger.warning(f"Graph search failed, falling back to dense retrieval: {str(e)}")
            
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