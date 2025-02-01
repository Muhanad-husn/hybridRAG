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
import traceback

# Get logger instance
logger = logging.getLogger(__name__)

class HybridRetrieval:
    """Implements hybrid retrieval combining graph-based and embedding-based search."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the hybrid retrieval system."""
        self.config = self._load_config(config_path)
        self.ranker = self._initialize_ranker()
        
        # Initialize embedding generator (singleton)
        logger.info("Initializing embedding generator...")
        self.embedding_generator = EmbeddingGenerator(config_path)
        
        # Initialize vector store as None (lazy initialization)
        self.vector_store = None
        
        # Initialize confidence calculation cache
        self.confidence_cache = {}

    def _initialize_vector_store(self):
        """Initialize or load the vector store."""
        if self.vector_store is None:
            embeddings_dir = os.path.join('data', 'embeddings')
            index_path = os.path.join(embeddings_dir, 'index.faiss')
            
            if not os.path.exists(embeddings_dir):
                os.makedirs(embeddings_dir)
            
            if os.path.exists(index_path):
                logger.info(f"Loading existing FAISS index from {index_path}")
                self.vector_store = self.embedding_generator.vector_store
            else:
                logger.info("Creating new empty vector store")
                self.vector_store = self.embedding_generator.create_empty_vector_store()
            
            logger.info("Vector store initialized")
        
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
            # Ensure vector store is initialized
            self._initialize_vector_store()
            
            # Add documents to the vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to the vector store")
            
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
            # Ensure vector store is initialized
            self._initialize_vector_store()
            
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
                    total_docs = len(self.vector_store.docstore._dict)
                    for idx, score in zip(indices[0], scores[0]):
                        if idx != -1 and idx < total_docs:  # Valid index within bounds
                            try:
                                doc_id = str(idx)
                                if doc_id in self.vector_store.docstore._dict:
                                    doc = self.vector_store.docstore._dict[doc_id]
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
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
                
                if not results:
                    logger.warning("No results found with similarity_search_with_score, trying similarity_search")
                    docs = self.vector_store.similarity_search(
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

    # Removed old _apply_diversity_penalty method

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
            passages = self._prepare_passages(results)
            
            if not passages:  # Handle empty results
                return []

            if len(passages) == 1:  # Handle single result
                return [self._format_single_result(passages[0])]

            # Rerank passages
            reranked_results = self._rerank_passages(query, passages)
            
            # Apply diversity penalty and sort
            diversity_adjusted_results = self._apply_diversity_penalty(reranked_results)
            
            # Sort and apply top_k
            sorted_results = sorted(diversity_adjusted_results, key=lambda x: x['score'], reverse=True)
            sorted_results = sorted_results[:top_k] if top_k else sorted_results
            
            # Log diversity metrics
            self._log_diversity_metrics(sorted_results)
            
            # Format results
            return self._format_results(sorted_results)
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            raise

    def _prepare_passages(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                passage = self._prepare_graph_passage(result, idx)
            else:  # Handle dictionary format
                passage = {
                    "id": idx,
                    "text": result.get("text", ""),
                    "meta": result.get("meta", "unknown"),
                    "score": float(result.get("score", 0.0))
                }
            passages.append(passage)
        return passages

    def _prepare_graph_passage(self, result: Dict[str, Any], idx: int) -> Dict[str, Any]:
        node = result['node']
        props = node.get('properties', {})
        if isinstance(props, str):
            try:
                import json
                props = json.loads(props)
            except:
                props = {'text': props}
        
        text_parts = [
            f"Type: {node.get('type', 'unknown')}",
            *[f"{k}: {v}" for k, v in props.items() if k != 'source']
        ]
        
        return {
            "id": idx,
            "text": "\n".join(text_parts),
            "meta": node.get('id', 'unknown'),
            "score": float(node.get('relevance_score', 0.0))
        }

    def _format_single_result(self, passage: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": passage["id"],
            "text": passage["text"],
            "meta": passage["meta"],
            "score": 1.0
        }

    def _rerank_passages(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = [[query, passage["text"]] for passage in passages]
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**features)
            logits = outputs.logits
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            scores = torch.nn.functional.softmax(logits, dim=1)
            scores = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
            scores = scores.cpu().numpy()

        return [
            {
                "id": passage["id"],
                "text": passage["text"],
                "meta": passage["meta"],
                "score": float(score)
            }
            for passage, score in zip(passages, scores)
        ]

    def _apply_diversity_penalty(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        source_counts = {}
        diversity_adjusted_results = []
        
        for result in results:
            source_type = self._get_source_type(result)
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            adjusted_score = self._calculate_diversity_penalty(
                result['score'],
                source_type,
                source_counts
            )
            
            result['original_score'] = result['score']
            result['score'] = adjusted_score
            diversity_adjusted_results.append(result)
        
        return diversity_adjusted_results

    def _calculate_diversity_penalty(self, score: float, source_type: str, source_counts: Dict[str, int]) -> float:
        """Calculate diversity penalty based on source type frequency."""
        config = self.config.get('diversity', {})
        if not config.get('enable_penalty', True):
            return score
            
        penalty_factor = config.get('penalty_factor', 0.05)
        max_penalty = config.get('max_penalty', 0.3)
        
        # Calculate penalty based on how many times this source type has been seen
        penalty = min(source_counts[source_type] * penalty_factor, max_penalty)
        return score * (1 - penalty)

    def _log_diversity_metrics(self, results: List[Dict[str, Any]]) -> None:
        final_source_counts = {}
        for result in results:
            source_type = self._get_source_type(result)
            final_source_counts[source_type] = final_source_counts.get(source_type, 0) + 1
        
        logger.info(f"Source type distribution in final results: {final_source_counts}")
        coverage = len(final_source_counts) / self.config.get('diversity', {}).get('min_source_types', 3)
        logger.info(f"Source type coverage: {coverage:.2%}")

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_results = []
        for result in results:
            confidence = self._calculate_confidence(result)
            formatted_result = {
                "text": result["text"],
                "meta": result["meta"],
                "score": float(result["score"]),
                "page_content": result["text"],
                "metadata": {"source": result["meta"]},
                "confidence": confidence
            }
            formatted_results.append(formatted_result)
        return formatted_results

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate and cache confidence score for a result."""
        cache_key = (result["text"], result["meta"], result["score"])
        if cache_key in self.confidence_cache:
            return self.confidence_cache[cache_key]

        # Calculate confidence score
        relevance_score = float(result["score"])
        source_coverage = self._calculate_source_coverage(result)
        answer_completeness = self._calculate_answer_completeness(result)

        confidence = (relevance_score + source_coverage + answer_completeness) / 3
        
        # Cache the calculated confidence
        self.confidence_cache[cache_key] = confidence
        
        return confidence

    def _calculate_source_coverage(self, result: Dict[str, Any]) -> float:
        """Calculate source coverage based on the diversity of sources."""
        source_type = self._get_source_type(result)
        source_counts = self.config.get('diversity', {}).get('source_counts', {})
        total_sources = sum(source_counts.values())
        if total_sources == 0:
            return 0.5  # Default value if no sources are available
        
        source_ratio = source_counts.get(source_type, 0) / total_sources
        return 1 - source_ratio  # Higher score for less common sources

    def _calculate_answer_completeness(self, result: Dict[str, Any]) -> float:
        """Calculate answer completeness based on text length and content."""
        text = result.get("text", "")
        word_count = len(text.split())
        
        # Assume optimal answer length is between 50 and 200 words
        if word_count < 50:
            completeness = word_count / 50
        elif word_count > 200:
            completeness = 1 - ((word_count - 200) / 200)
        else:
            completeness = 1.0
        
        # Check for key phrases that might indicate a complete answer
        key_phrases = ["in conclusion", "to summarize", "therefore"]
        if any(phrase in text.lower() for phrase in key_phrases):
            completeness += 0.1
        
        return min(max(completeness, 0), 1)  # Ensure the score is between 0 and 1

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
        combined_results = []
        
        try:
            # Get dense retrieval results
            logger.info(f"Processing dense retrieval results in {mode} mode")
            embedding_results = self.similarity_search(query, k=top_k)
            
            if not embedding_results:
                logger.warning("No dense retrieval results found")
                return []

            # Process dense retrieval results
            for doc, score in embedding_results:
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    result = {
                        'text': doc.page_content,
                        'meta': doc.metadata.get('source', 'unknown'),
                        'score': float(score),
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    result['confidence'] = self._calculate_confidence(result)
                    combined_results.append(result)

            logger.info(f"Added {len(combined_results)} valid dense retrieval results")

            # Add graph-based results if in hybrid mode
            if mode == "Hybrid" and graph is not None:
                logger.info(f"Processing graph with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
                min_degree = self.config.get('graph', {}).get('min_degree', 2)
                relationships = self._process_graph_relationships(graph, min_degree)
                
                if relationships:
                    graph_results = self._group_relationships(relationships)
                    for result in graph_results:
                        result['confidence'] = self._calculate_confidence(result)
                    combined_results.extend(graph_results)

            # Deduplicate and sort results
            unique_results = self.deduplicate_results(combined_results)
            sorted_results = sorted(unique_results, key=lambda x: float(x.get('confidence', 0.0)), reverse=True)
            
            # Apply top_k if specified
            final_results = sorted_results[:rerank_top_k] if rerank_top_k else sorted_results
            
            # Format final results
            return self._format_results(final_results)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            traceback.print_exc()
            return []

    def _process_graph_relationships(self, graph: GraphConstructor, min_degree: int) -> List[Dict[str, Any]]:
        """Process graph relationships."""
        relationships = []
        high_degree_nodes = 0

        for node in graph.graph.nodes():
            degree = graph.graph.degree(node)
            if degree > min_degree:
                high_degree_nodes += 1
                node_data = graph.graph.nodes[node]
                node_type = node_data.get('type', 'unknown')
                
                for neighbor in graph.graph.neighbors(node):
                    edge = graph.graph.get_edge_data(node, neighbor)
                    if edge:
                        rel_type = edge.get('type', 'unknown')
                        rel_score = min(0.9, 0.7 + (degree - min_degree) * 0.05)
                        relationships.append({
                            'text': f"{node} ({node_type}) -{rel_type}-> {neighbor}",
                            'score': rel_score,
                            'type': rel_type
                        })

        logger.info(f"Found {high_degree_nodes} high-degree nodes with {len(relationships)} relationships")
        return relationships

    def _group_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group relationships by type."""
        rel_by_type = {}
        for rel in relationships:
            rel_type = rel['type']
            if rel_type not in rel_by_type:
                rel_by_type[rel_type] = []
            rel_by_type[rel_type].append(rel)

        grouped_results = []
        for rel_type, rels in rel_by_type.items():
            sorted_rels = sorted(rels, key=lambda x: x['score'], reverse=True)
            rel_texts = [r['text'] for r in sorted_rels]
            graph_context = f"Graph Analysis ({rel_type}):\n" + "\n".join(rel_texts)
            group_score = sorted_rels[0]['score'] if sorted_rels else 0.7
            
            grouped_results.append({
                'text': graph_context,
                'meta': f'graph_relationships_{rel_type}',
                'score': group_score,
                'source_type': 'relationship'
            })

        logger.info(f"Added {len(rel_by_type)} relationship groups to results")
        return grouped_results

    def reset_vector_store(self) -> None:
        """Reset the vector store to an empty state."""
        try:
            # Call the reset_vector_store method of the embedding generator
            self.embedding_generator.reset_vector_store()
            
            # Update the vector store reference
            self.vector_store = self.embedding_generator.vector_store
            logger.info("Reset vector store to empty state")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            raise