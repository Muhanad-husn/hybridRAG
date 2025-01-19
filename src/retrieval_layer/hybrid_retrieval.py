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

# Get logger instance
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
            # Use absolute path for cache directory
            cache_dir = os.path.join(os.getcwd(), "data", "cache", "reranker")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize ranker with model from config
            model_name = self.config["ranking"].get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info(f"Initializing ranker with model: {model_name}")
            
            return Ranker(
                model_name=model_name,
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
        query: str,
        k: int = 100
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using Chroma vector store.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Use Chroma's similarity search
            results = self.embedding_generator.vector_store.similarity_search_with_score(
                query,
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
        Perform query-relevant graph-based search using NetworkX.
        
        Args:
            graph: GraphConstructor instance
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of graph search results ordered by relevance to query
        """
        try:
            import networkx as nx
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            results = []
            
            # Create text content for each node by combining its properties
            node_texts = {}
            for node, data in graph.graph.nodes(data=True):
                # Combine node ID, type and properties into searchable text
                node_type = data.get('type', 'unknown')
                properties = data.get('properties', {})
                if isinstance(properties, str):
                    # Handle JSON string properties
                    import json
                    try:
                        properties = json.loads(properties)
                    except:
                        properties = {'text': properties}
                
                # Create searchable text from node data
                text_parts = [
                    str(node),  # node ID
                    node_type,  # node type
                    *[str(v) for v in properties.values()]  # property values
                ]
                node_texts[node] = ' '.join(text_parts)
            
            if not node_texts:
                return []
            
            # Calculate text similarity between query and nodes
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
            try:
                # Create document matrix from node texts
                node_text_list = list(node_texts.values())
                text_matrix = vectorizer.fit_transform(node_text_list)
                # Transform query
                query_vector = vectorizer.transform([query])
                # Calculate similarities and normalize to [0,1] range
                similarities = cosine_similarity(query_vector, text_matrix)[0]
                if len(similarities) > 0:
                    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-10)
            except Exception as e:
                logger.warning(f"Text similarity calculation failed: {str(e)}")
                similarities = np.ones(len(node_texts)) / len(node_texts)
            
            # Calculate node importance using centrality measures
            try:
                # Calculate centrality measures
                degree_cent = nx.degree_centrality(graph.graph)
                between_cent = nx.betweenness_centrality(graph.graph)
                eigen_cent = nx.eigenvector_centrality_numpy(graph.graph)
                
                # Combine centrality measures with weights
                centrality_scores = {}
                for node in graph.graph.nodes():
                    centrality_scores[node] = (
                        0.4 * degree_cent[node] +
                        0.3 * between_cent[node] +
                        0.3 * eigen_cent[node]
                    )
                
                # Normalize centrality scores to [0,1]
                max_cent = max(centrality_scores.values())
                min_cent = min(centrality_scores.values())
                for node in centrality_scores:
                    centrality_scores[node] = (centrality_scores[node] - min_cent) / (max_cent - min_cent + 1e-10)
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {str(e)}")
                centrality_scores = {node: 1.0/len(graph.graph) for node in graph.graph.nodes()}
            
            # Combine text similarity with graph centrality
            node_scores = {}
            for i, node in enumerate(node_texts.keys()):
                text_score = float(similarities[i])
                centrality_score = float(centrality_scores[node])
                # Weighted combination of scores with higher weight on text similarity
                node_scores[node] = (0.7 * text_score + 0.3 * centrality_score)
            
            # Sort nodes by combined score
            sorted_nodes = sorted(
                node_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            # Extract relevant subgraph
            relevant_nodes = [node for node, _ in sorted_nodes]
            subgraph = graph.graph.subgraph(relevant_nodes)
            
            # Convert results to output format
            for node in relevant_nodes:
                node_data = graph.graph.nodes[node]
                
                # Get edges to other relevant nodes
                edges = []
                for neighbor in subgraph.neighbors(node):
                    edge_data = graph.graph.get_edge_data(node, neighbor)
                    if edge_data:  # Only include edges with data
                        edges.append({
                            'source': node,
                            'target': neighbor,
                            'type': edge_data.get('type', 'unknown'),
                            'properties': edge_data
                        })
                
                # Format node properties
                properties = node_data.copy()
                node_type = properties.pop('type', 'unknown')
                
                # Handle string properties
                if isinstance(properties.get('properties'), str):
                    try:
                        import json
                        properties = json.loads(properties['properties'])
                    except:
                        properties = {'text': properties['properties']}
                
                # Add node and its edges to results with formatted properties
                results.append({
                    'node': {
                        'id': node,
                        'type': node_type,
                        'properties': properties,
                        'relevance_score': float(node_scores[node])  # Include normalized relevance score
                    },
                    'edges': edges
                })
            
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
                    # Format node properties for better readability
                    props = node.get('properties', {})
                    if isinstance(props, str):
                        try:
                            import json
                            props = json.loads(props)
                        except:
                            props = {'text': props}
                    
                    # Create readable text from properties
                    text_parts = [
                        f"Type: {node.get('type', 'unknown')}",
                        *[f"{k}: {v}" for k, v in props.items() if k != 'source']
                    ]
                    
                    passage = {
                        "id": idx,
                        "text": "\n".join(text_parts),
                        "meta": node.get('id', 'unknown'),
                        "score": float(node.get('relevance_score', 0.0))  # Use relevance score from graph search
                    }
                else:  # Handle dictionary format
                    passage = {
                        "id": idx,
                        "text": result.get("text", ""),
                        "meta": result.get("meta", "unknown"),
                        "score": float(result.get("score", 0.0))
                    }
                passages.append(passage)
            
            # Convert passages to the format expected by FlashRank
            formatted_passages = []
            for idx, result in enumerate(results, start=1):
                if isinstance(result, tuple):  # Handle (doc, score) format
                    doc, score = result
                    formatted_passages.append({
                        "id": idx,
                        "text": doc.page_content,
                        "meta": doc.metadata
                    })
                elif 'node' in result:  # Handle graph search results
                    node = result['node']
                    props = node.get('properties', {})
                    if isinstance(props, str):
                        try:
                            import json
                            props = json.loads(props)
                        except:
                            props = {'text': props}
                    
                    # Create readable text from properties
                    text_parts = [
                        f"Type: {node.get('type', 'unknown')}",
                        *[f"{k}: {v}" for k, v in props.items() if k != 'source']
                    ]
                    formatted_passages.append({
                        "id": idx,
                        "text": "\n".join(text_parts),
                        "meta": {"id": node.get('id', 'unknown')}
                    })
                else:  # Handle dictionary format
                    formatted_passages.append({
                        "id": idx,
                        "text": result.get("text", ""),
                        "meta": result.get("meta", {})
                    })

            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=formatted_passages)
            
            # Perform reranking
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
            query_embedding: Query embedding vector (not used directly, we use query text instead)
            graph: GraphConstructor instance
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after reranking
            mode: Search mode ("Hybrid" or "Dense")
            
        Returns:
            Combined and ranked search results
        """
        # Initialize results
        combined_results = []
        
        try:
            # First get dense retrieval results
            logger.info(f"Processing dense retrieval results in {mode} mode")
            embedding_results = self.similarity_search(query, k=top_k)
            
            if embedding_results:
                for doc, score in embedding_results:
                    combined_results.append({
                        'text': doc.page_content,
                        'meta': doc.metadata.get('source', 'unknown'),
                        'score': float(score)
                    })
                logger.info(f"Added {len(embedding_results)} dense retrieval results")

            # Then add graph-based results if in hybrid mode
            if mode == "Hybrid" and graph is not None:
                logger.info(f"Processing graph with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
                # Find important relationships in the graph
                relationships = []
                high_degree_nodes = 0
                
                # Get nodes with many relationships (similar to Neo4j COUNT query)
                for node in graph.graph.nodes():
                    degree = graph.graph.degree(node)
                    if degree > 3:
                        high_degree_nodes += 1
                        node_data = graph.graph.nodes[node]
                        node_type = node_data.get('type', 'unknown')
                        logger.info(f"Found high-degree node: {node} ({node_type}) with {degree} connections")
                        
                        # Get all relationships for this node
                        for neighbor in graph.graph.neighbors(node):
                            edge = graph.graph.get_edge_data(node, neighbor)
                            if edge:
                                rel_type = edge.get('type', 'unknown')
                                relationships.append(f"{node} ({node_type}) -{rel_type}-> {neighbor}")
                
                logger.info(f"Found {high_degree_nodes} high-degree nodes with {len(relationships)} relationships")
                
                if relationships:
                    # Create a document from graph relationships
                    graph_context = "\n".join(relationships)
                    combined_results.append({
                        'text': f"Graph Analysis:\n{graph_context}",
                        'meta': 'graph_relationships',
                        'score': 0.9  # High score for graph-based context
                    })
                    logger.info("Added graph relationships to results")

            # Deduplicate results
            unique_results = self.deduplicate_results(combined_results)
            
            # Sort by score in descending order
            sorted_results = sorted(
                unique_results,
                key=lambda x: float(x.get('score', 0.0)),
                reverse=True
            )
            
            # Apply top_k if specified
            if rerank_top_k:
                sorted_results = sorted_results[:rerank_top_k]
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            traceback.print_exc()
            raise