import os
import logging
import shutil
import pandas as pd
from typing import List, Dict, Any, Optional
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.utils.logger import setup_logger

class HyperRAG:
    """Main class for the Hyper RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Hyper RAG system."""
        # Setup logging
        setup_logger()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.document_processor = DocumentProcessor(config_path)
        self.embedding_generator = EmbeddingGenerator(config_path)
        
        # Try to initialize graph constructor (optional)
        try:
            self.graph_constructor = GraphConstructor(config_path)
        except Exception as e:
            self.logger.warning(f"Graph constructor initialization failed: {str(e)}")
            self.graph_constructor = None
            
        self.retrieval_system = HybridRetrieval(config_path)
        
        self.logger.info("Hyper RAG system initialized")

    def reset_storage(self) -> None:
        """Reset all storage (vector store and graph files)."""
        try:
            # Reset vector store
            try:
                self.embedding_generator.reset_vector_store()
                self.logger.info("Reset vector store")
            except Exception as e:
                self.logger.error(f"Error resetting vector store: {str(e)}")
                raise

            # Reset graph files
            graphs_dir = os.path.join('data', 'graphs')
            if os.path.exists(graphs_dir):
                shutil.rmtree(graphs_dir)
            os.makedirs(graphs_dir)
            # Initialize empty CSV files with headers
            nodes_file = os.path.join(graphs_dir, 'nodes.csv')
            edges_file = os.path.join(graphs_dir, 'edges.csv')
            pd.DataFrame(columns=['id', 'type', 'properties']).to_csv(nodes_file, index=False)
            pd.DataFrame(columns=['source_id', 'target_id', 'type', 'properties']).to_csv(edges_file, index=False)
            self.logger.info(f"Reset and initialized graph files at {graphs_dir}")

            # Reset processed chunks
            chunks_dir = os.path.join('data', 'processed_chunks')
            if os.path.exists(chunks_dir):
                shutil.rmtree(chunks_dir)
                os.makedirs(chunks_dir)
                self.logger.info(f"Reset processed chunks at {chunks_dir}")

            self.logger.info("Storage reset completed")
        except Exception as e:
            self.logger.error(f"Error resetting storage: {str(e)}")
            raise

    def process_documents(
        self,
        input_dir: str,
        save_chunks: bool = True,
        save_embeddings: bool = True
    ) -> None:
        """
        Process documents from the input directory.
        
        Args:
            input_dir: Directory containing input documents
            save_chunks: Whether to save processed chunks
            save_embeddings: Whether to save generated embeddings
        """
        try:
            self.logger.info(f"Processing documents from {input_dir}")
            
            # Process documents
            documents = self.document_processor.process_directory(input_dir)
            if not documents:
                raise ValueError("No documents were processed successfully")
            
            # Save chunks if requested
            if save_chunks:
                output_dir = os.path.join("data", "processed_chunks")
                self.document_processor.save_processed_chunks(documents, output_dir)
            
            # Generate embeddings
            documents = self.embedding_generator.process_documents(documents)
            
            # Save embeddings if requested
            if save_embeddings:
                self.embedding_generator.save_embeddings(documents)
            
            # Construct knowledge graph if available
            if self.graph_constructor is not None:
                try:
                    self.graph_constructor.construct_graph(documents)
                except Exception as e:
                    self.logger.warning(f"Graph construction failed: {str(e)}")
                
            # Build retrieval index (always required)
            self.retrieval_system.build_index(documents)
            
            self.logger.info("Document processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            raise

    def query(
        self,
        query: str,
        mode: str = "Hybrid",
        top_k: int = 100,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the system using the specified mode.
        
        Args:
            query: Search query
            mode: Search mode ("Hybrid" or "Dense")
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after reranking
            
        Returns:
            List of search results
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Perform search (hybrid if graph is available, dense otherwise)
            actual_mode = mode if self.graph_constructor is not None else "Dense"
            if actual_mode != mode:
                self.logger.warning(f"Falling back to {actual_mode} mode as graph is not available")
                
            results = self.retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=self.graph_constructor,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                mode=actual_mode
            )
            
            self.logger.info(f"Query processed successfully, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

def format_result(result: Dict[str, Any]) -> str:
    """Format a single search result for display."""
    if isinstance(result, tuple):
        # Handle document results from similarity search
        doc, score = result
        return (
            f"Document Chunk:\n"
            f"Content: {doc.page_content.strip()[:500]}...\n"
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Relevance: {score:.4f}"
        )
    elif 'node' in result:
        # Handle graph search results
        node = result['node']
        edges = result['edges']
        
        # Format node properties
        properties = node.get('properties', {})
        if isinstance(properties, str):
            # If properties is a string (e.g., from JSON), clean it up
            properties = properties.replace('{', '').replace('}', '').replace('"', '')
        property_text = ', '.join(f"{k}: {v}" for k, v in properties.items() if k != 'source')
        
        # Format edges
        edge_info = []
        for edge in edges:
            # Get target node properties if available
            target_props = edge.get('properties', {})
            if isinstance(target_props, str):
                target_props = target_props.replace('{', '').replace('}', '').replace('"', '')
            
            edge_info.append(
                f"- {edge['type']} -> {edge['target']} ({target_props})"
            )
        edge_summary = "\n".join(edge_info) if edge_info else "No connections"
        
        return (
            f"Entity: {node['id']}\n"
            f"Type: {node['type']}\n"
            f"Properties: {property_text}\n"
            f"Connections:\n{edge_summary}"
        )
    else:
        # Handle reranked results
        text = result.get('text', '')
        if isinstance(text, str):
            text = text.strip()[:500]  # Show more content, up to 500 chars
        meta = result.get('meta', 'unknown')
        score = result.get('score', 0.0)
        
        # Normalize score to 0-1 range if it's very small
        if score < 0.01:
            score = score * 100
        
        return (
            f"Content: {text}...\n"
            f"Source: {meta}\n"
            f"Relevance: {score:.4f}"
        )

def main():
    """Main entry point for the application."""
    try:
        # Setup logging
        setup_logger()
        logger = logging.getLogger(__name__)
        
        # Initialize the system
        rag_system = HyperRAG()
        
        # Reset storage to clear existing state
        rag_system.reset_storage()
        logger.info("Reset storage to clear existing state")
        
        # Example usage
        input_dir = os.path.join("data", "raw_documents")
        
        # Process documents (this will trigger LLM extraction and graph construction)
        rag_system.process_documents(
            input_dir=input_dir,
            save_chunks=True,
            save_embeddings=True
        )
        
        # Test query
        query = "What were the key factors that escalated the peaceful demonstrations in Syria into a full-scale armed conflict?"
        
        print(f"\nQuery:")
        print(query)
        
        # Perform hybrid search with reranking
        print("\nHybrid Search Results:")
        results = rag_system.query(
            query=query,
            mode="Hybrid",
            top_k=10,  # Get more results initially
            rerank_top_k=5  # Rerank and limit to top 5
        )
        
        # Print results
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(format_result(result))
            print("-" * 80)  # Add separator between results
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()