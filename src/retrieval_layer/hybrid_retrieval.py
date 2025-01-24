import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        self.embedding_generator = EmbeddingGenerator(config_path)
        
        # Verify vector store exists
        embeddings_dir = os.path.join('data', 'embeddings')
        index_path = os.path.join(embeddings_dir, 'index.faiss')
        if not os.path.exists(index_path):
            raise ValueError("Vector store not initialized. Please run src/main.py first to initialize the system.")
        logger.info(f"Found existing FAISS index at {index_path}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _initialize_ranker(self):
        """Initialize the reranking model using transformers."""
        try:
            model_name = self.config["ranking"]["model_name"]
            logger.info(f"Initializing ranker with model: {model_name}")
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
        except Exception as e:
            logger.error(f"Error initializing ranker: {str(e)}")
            raise

    def build_index(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with embeddings in metadata
        """
        try:
            # Skip building index since save_embeddings already handles this
            logger.info("Skipping index build - already handled by save_embeddings")
            
        except Exception as e:
            logger.error(f"Error in build_index: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        # Use configured pool size or default to 100
        k = k or self.config.get('retrieval', {}).get('initial_pool_size', 100)
        """
        Perform similarity search using FAISS vector store with inner product similarity.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Get raw index for direct search if available
            raw_index = getattr(self.embedding_generator, 'raw_index', None)
            
            if raw_index is not None:
                try:
                    # Generate query embedding
                    query_embedding = self.embedding_generator.generate_embedding(query)
                    
                    # Normalize query vector for inner product search
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    
                    # Perform search with raw index
                    scores, indices = raw_index.search(
                        query_embedding.reshape(1, -1).astype('float32'),
                        k
                    )
                    
                    # Log search statistics
                    logger.info(f"Raw search found {len(indices[0])} results")
                    logger.info(f"Score range: {scores[0].min():.4f} to {scores[0].max():.4f}")
                    
                    # Get documents from vector store
                    docs = []
                    total_docs = len(self.embedding_generator.vector_store.docstore._dict)
                    for idx, score in zip(indices[0], scores[0]):
                        if idx != -1 and idx < total_docs:  # Valid index within bounds
                            try:
                                doc_id = str(idx)
                                if doc_id in self.embedding_generator.vector_store.docstore._dict:
                                    doc = self.embedding_generator.vector_store.docstore._dict[doc_id]
                                    docs.append((doc, float(score)))
                            except KeyError:
                                logger.warning(f"Document not found for index {idx}")
                                continue
                    
                    if docs:
                        logger.info(f"Retrieved {len(docs)} valid documents")
                        return docs
                    else:
                        logger.warning("No valid documents found in raw search")
                
                except Exception as e:
                    logger.error(f"Error in raw index search: {str(e)}")
                    # Fall back to standard search
            
            # Standard vector store search as fallback
            try:
                results = self.embedding_generator.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
                
                if not results:
                    logger.warning("No results found with similarity_search_with_score, trying similarity_search")
                    docs = self.embedding_generator.vector_store.similarity_search(
                        query,
                        k=k
                    )
                    if docs:
                        results = [(doc, 0.9) for doc in docs]
                    else:
                        logger.warning("No results found in vector store")
                        return []
                
                logger.info(f"Found {len(results)} results in vector store")
                
                # Format results
                formatted_results = []
                for doc, score in results:
                    if hasattr(doc, 'page_content'):
                        formatted_results.append((doc, float(score)))
                    else:
                        formatted_doc = Document(
                            page_content=str(doc),
                            metadata={'source': 'unknown'}
                        )
                        formatted_results.append((formatted_doc, float(score)))
                return formatted_results
                
            except Exception as e:
                logger.error(f"Error in vector store search: {str(e)}\n{traceback.format_exc()}")
                return []
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def graph_search(
        self,
        graph: GraphConstructor,
        query: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        # Use configured limit or default
        limit = limit or self.config.get('graph', {}).get('node_limit', 150)
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
                # Use configured weights or defaults
                text_weight = self.config.get('graph', {}).get('text_weight', 0.6)
                centrality_weight = self.config.get('graph', {}).get('centrality_weight', 0.4)
                node_scores[node] = (text_weight * text_score + centrality_weight * centrality_score)
            
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

    def _get_source_type(self, result: Dict[str, Any]) -> str:
        """Determine the source type of a result."""
        if isinstance(result, dict):
            if 'node' in result:
                return 'graph'
            if 'meta' in result and isinstance(result['meta'], str):
                if 'graph_relationships' in result['meta']:
                    return 'relationship'
                if result['meta'].endswith(('.pdf', '.docx', '.txt')):
                    return 'document'
            return 'unknown'
        return 'unknown'

    def _apply_diversity_penalty(self, score: float, source_type: str, source_counts: Dict[str, int]) -> float:
        """Apply diversity penalty based on source type frequency."""
        config = self.config.get('diversity', {})
        if not config.get('enable_penalty', True):
            return score
            
        penalty_factor = config.get('penalty_factor', 0.05)
        max_penalty = config.get('max_penalty', 0.3)
        
        # Calculate penalty based on how many times this source type has been seen
        penalty = min(source_counts[source_type] * penalty_factor, max_penalty)
        return score * (1 - penalty)

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        # Use configured rerank size if top_k not specified
        top_k = top_k or self.config.get('retrieval', {}).get('rerank_size', 50)
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
            
            if not passages:  # Handle empty results
                return []

            if len(passages) == 1:  # Handle single result
                return [{
                    "id": passages[0]["id"],
                    "text": passages[0]["text"],
                    "meta": passages[0]["meta"],
                    "score": 1.0
                }]

            # Prepare inputs for the model
            pairs = [[query, passage["text"]] for passage in passages]

            # Tokenize all pairs
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Get scores from model
            with torch.no_grad():
                outputs = self.model(**features)
                # Handle both single and batch predictions
                logits = outputs.logits
                if logits.dim() == 1:
                    # Single prediction case - reshape to [1, num_classes]
                    logits = logits.unsqueeze(0)
                scores = torch.nn.functional.softmax(logits, dim=1)
                # Ensure we have the right dimension before indexing
                if scores.shape[1] > 1:
                    scores = scores[:, 1]
                else:
                    scores = scores[:, 0]
                scores = scores.cpu().numpy()

            # Create reranked results
            reranked_results = [
                {
                    "id": passage["id"],
                    "text": passage["text"],
                    "meta": passage["meta"],
                    "score": float(score)
                }
                for passage, score in zip(passages, scores)
            ]
            
            # Track source types and apply diversity penalty
            source_counts = {}
            diversity_adjusted_results = []
            
            for result in reranked_results:
                source_type = self._get_source_type(result)
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
                
                # Apply diversity penalty to score
                adjusted_score = self._apply_diversity_penalty(
                    result['score'],
                    source_type,
                    source_counts
                )
                
                result['original_score'] = result['score']
                result['score'] = adjusted_score
                diversity_adjusted_results.append(result)
            
            # Sort by adjusted score
            sorted_results = sorted(
                diversity_adjusted_results,
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Apply top_k and log diversity metrics
            if top_k:
                sorted_results = sorted_results[:top_k]
                
            # Log diversity metrics
            final_source_counts = {}
            for result in sorted_results:
                source_type = self._get_source_type(result)
                final_source_counts[source_type] = final_source_counts.get(source_type, 0) + 1
            
            logger.info(f"Source type distribution in final results: {final_source_counts}")
            coverage = len(final_source_counts) / self.config.get('diversity', {}).get('min_source_types', 3)
            logger.info(f"Source type coverage: {coverage:.2%}")
            
            # Convert to standard format with both old and new fields
            formatted_results = []
            for result in sorted_results:
                formatted_result = {
                    "text": result["text"],
                    "meta": result["meta"],
                    "score": float(result["score"]),
                    # Add page_content for compatibility
                    "page_content": result["text"],
                    # Add metadata for compatibility
                    "metadata": {"source": result["meta"]}
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
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
            try:
                # First get dense retrieval results
                logger.info(f"Processing dense retrieval results in {mode} mode")
                embedding_results = self.similarity_search(query, k=top_k)
                
                if embedding_results:
                    for doc, score in embedding_results:
                        # Ensure we have valid content
                        if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                            logger.warning(f"Skipping invalid document: {doc}")
                            continue
                            
                        # Format the result with both old and new fields
                        result = {
                            'text': doc.page_content,
                            'meta': doc.metadata.get('source', 'unknown'),
                            'score': float(score),
                            'page_content': doc.page_content,
                            'metadata': doc.metadata
                        }
                        combined_results.append(result)
                    
                    if combined_results:
                        logger.info(f"Added {len(combined_results)} valid dense retrieval results")
                    else:
                        logger.warning("No valid results after filtering")
                        return []
                else:
                    logger.warning("No dense retrieval results found")
                    return []  # Return empty list if no results found
            except Exception as e:
                logger.error(f"Error in dense retrieval: {str(e)}\n{traceback.format_exc()}")
                return []  # Return empty list on error

            # Then add graph-based results if in hybrid mode
            if mode == "Hybrid" and graph is not None:
                logger.info(f"Processing graph with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
                # Find important relationships in the graph
                relationships = []
                high_degree_nodes = 0
                
                # Get minimum degree from config
                min_degree = self.config.get('graph', {}).get('min_degree', 2)
                
                # Get nodes with significant relationships
                for node in graph.graph.nodes():
                    degree = graph.graph.degree(node)
                    if degree > min_degree:
                        high_degree_nodes += 1
                        node_data = graph.graph.nodes[node]
                        node_type = node_data.get('type', 'unknown')
                        logger.debug(f"Found high-degree node: {node} ({node_type}) with {degree} connections")
                        
                        # Get all relationships for this node
                        for neighbor in graph.graph.neighbors(node):
                            edge = graph.graph.get_edge_data(node, neighbor)
                            if edge:
                                rel_type = edge.get('type', 'unknown')
                                # Dynamic scoring based on degree and relationship type
                                rel_score = min(0.9, 0.7 + (degree - min_degree) * 0.05)
                                relationships.append({
                                    'text': f"{node} ({node_type}) -{rel_type}-> {neighbor}",
                                    'score': rel_score,
                                    'type': rel_type
                                })
                
                logger.info(f"Found {high_degree_nodes} high-degree nodes with {len(relationships)} relationships")
                
                if relationships:
                    # Group relationships by type for better organization
                    rel_by_type = {}
                    for rel in relationships:
                        rel_type = rel['type']
                        if rel_type not in rel_by_type:
                            rel_by_type[rel_type] = []
                        rel_by_type[rel_type].append(rel)
                    
                    # Add each relationship group as a separate result
                    for rel_type, rels in rel_by_type.items():
                        # Sort relationships by score
                        sorted_rels = sorted(rels, key=lambda x: x['score'], reverse=True)
                        
                        # Create context with relationships of this type
                        rel_texts = [r['text'] for r in sorted_rels]
                        graph_context = f"Graph Analysis ({rel_type}):\n" + "\n".join(rel_texts)
                        
                        # Use maximum relationship score for this group
                        group_score = sorted_rels[0]['score'] if sorted_rels else 0.7
                        
                        combined_results.append({
                            'text': graph_context,
                            'meta': f'graph_relationships_{rel_type}',
                            'score': group_score,
                            'source_type': 'relationship'
                        })
                    
                    logger.info(f"Added {len(rel_by_type)} relationship groups to results")

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