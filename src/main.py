import os
import logging
from typing import List, Dict, Any, Optional
from input_layer.document_processor import DocumentProcessor
from processing_layer.embedding_generator import EmbeddingGenerator
from processing_layer.graph_constructor import GraphConstructor
from retrieval_layer.hybrid_retrieval import HybridRetrieval
from utils.logger import setup_logger

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
        self.graph_constructor = GraphConstructor(config_path)
        self.retrieval_system = HybridRetrieval(config_path)
        
        self.logger.info("Hyper RAG system initialized")

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
                output_dir = os.path.join("data", "embeddings")
                self.embedding_generator.save_embeddings(documents, output_dir)
            
            # Construct knowledge graph
            self.graph_constructor.construct_graph(documents)
            
            # Build retrieval index
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
            
            # Perform hybrid search
            results = self.retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph_db=self.graph_constructor.graph,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                mode=mode
            )
            
            self.logger.info(f"Query processed successfully, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    def clear_graph(self) -> None:
        """Clear the knowledge graph."""
        try:
            self.graph_constructor.clear_graph()
            self.logger.info("Knowledge graph cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing graph: {str(e)}")
            raise

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
        
        # Example query
        query = "What are the main topics discussed in the documents?"
        results = rag_system.query(
            query=query,
            mode="Hybrid",
            top_k=10
        )
        
        # Print results
        print("\nSearch Results:")
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Source: {result['meta']}")
            print(f"Score: {result['score']:.4f}")
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()