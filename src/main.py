import os
import logging
import shutil
from typing import List, Dict, Any, Optional
from .input_layer.document_processor import DocumentProcessor
from .processing_layer.embedding_generator import EmbeddingGenerator
from .processing_layer.graph_constructor import GraphConstructor
from .retrieval_layer.hybrid_retrieval import HybridRetrieval
from .utils.logger import setup_logger

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
            if self.graph_constructor is not None:
                graphs_dir = os.path.join('data', 'graphs')
                if os.path.exists(graphs_dir):
                    shutil.rmtree(graphs_dir)
                    os.makedirs(graphs_dir)
                    self.logger.info(f"Reset graph files at {graphs_dir}")

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
            
            # Reset storage before processing
            self.reset_storage()
            
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

    def clear_graph(self) -> None:
        """Clear the knowledge graph."""
        try:
            if self.graph_constructor is not None:
                self.graph_constructor.clear_graph()
                self.logger.info("Knowledge graph cleared successfully")
            else:
                self.logger.warning("No graph constructor available to clear")
        except Exception as e:
            self.logger.error(f"Error clearing graph: {str(e)}")
            raise

def format_result(result: Dict[str, Any]) -> str:
    """Format a single search result for display."""
    if isinstance(result, tuple):
        # Handle document results from similarity search
        doc, score = result
        return (
            f"Document Chunk:\n"
            f"Content: {doc.page_content[:200]}...\n"
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Relevance: {score:.4f}"
        )
    elif 'node' in result:
        # Handle graph search results
        node = result['node']
        edges = result['edges']
        edge_info = []
        for edge in edges:
            edge_info.append(
                f"- {edge['type']} -> {edge['target']}"
            )
        edge_summary = "\n".join(edge_info) if edge_info else "No connections"
        
        return (
            f"Entity: {node['id']}\n"
            f"Type: {node['type']}\n"
            f"Connections:\n{edge_summary}"
        )
    else:
        # Handle reranked results
        text = result.get('text', '')[:200]
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
        # Initialize the system
        rag_system = HyperRAG()
        
        # Example usage
        input_dir = os.path.join("data", "raw_documents")
        
        # Process documents
        rag_system.process_documents(
            input_dir=input_dir,
            save_chunks=True,
            save_embeddings=True
        )
        
        # Test queries
        queries = [
            "What were the key factors that escalated the peaceful demonstrations in Syria into a full-scale armed conflict?",
            "How has the Syrian Civil War impacted the displacement of people both internally and internationally?",
            "What role has Russia played in shaping the dynamics of the Syrian Civil War, and how did its intervention affect the conflict?"
        ]
        
        # Process each query
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}:")
            print(query)
            print("\nResults:")
            
            results = rag_system.query(
                query=query,
                mode="Hybrid",
                top_k=5  # Limit to top 5 most relevant results
            )
            
            # Print results
            for idx, result in enumerate(results, 1):
                print(f"\nResult {idx}:")
                print(format_result(result))
                print("-" * 80)  # Add separator between results
            
            print("\n" + "=" * 100 + "\n")  # Add separator between queries
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()