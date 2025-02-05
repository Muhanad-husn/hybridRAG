import os
import logging
import shutil
from typing import Optional, Any
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.utils.error_handler import log_errors
from src.utils.config_handler import config

class HyperRAG:
    """Main class for the Hyper RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml", logger=None):
        """Initialize the Hyper RAG system."""
        # Use provided logger or get a new one
        self.logger = logger or logging.getLogger(__name__)
        
        # Load configuration
        config.load_config(config_path)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator(config_path)  # Will update in next phase
        
        # Try to initialize graph constructor (optional)
        try:
            self.graph_constructor = GraphConstructor(config_path)  # Will update in next phase
        except Exception as e:
            self.logger.warning(f"Graph constructor initialization failed: {str(e)}")
            self.graph_constructor = None
            
        self.retrieval_system = HybridRetrieval(config_path)  # Will update in next phase
        
        self.logger.info("Hyper RAG system initialized")

    @log_errors(logging.getLogger(__name__))
    def reset_storage(self) -> None:
        """Reset all storage (vector store and graph files)."""
        # Reset vector store
        self.embedding_generator.reset_vector_store()
        self.logger.info("Reset vector store")

        # Reset graph files
        graphs_dir = os.path.join('data', 'graphs')
        if os.path.exists(graphs_dir):
            shutil.rmtree(graphs_dir)
        os.makedirs(graphs_dir)
        
        # Initialize empty CSV files
        self._initialize_graph_files(graphs_dir)
        self.logger.info(f"Reset and initialized graph files at {graphs_dir}")

        # Reset processed chunks
        chunks_dir = os.path.join('data', 'processed_chunks')
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)
            os.makedirs(chunks_dir)
            self.logger.info(f"Reset processed chunks at {chunks_dir}")

        self.logger.info("Storage reset completed")

    def _initialize_graph_files(self, graphs_dir: str) -> None:
        """Initialize empty graph files."""
        nodes_file = os.path.join(graphs_dir, 'nodes.csv')
        edges_file = os.path.join(graphs_dir, 'edges.csv')
        
        with open(nodes_file, 'w') as f:
            f.write('id,type,properties\n')
        with open(edges_file, 'w') as f:
            f.write('source_id,target_id,type,properties\n')

    @log_errors(logging.getLogger(__name__))
    async def aprocess_documents(
        self,
        input_dir: str,
        save_chunks: bool = True,
        save_embeddings: bool = True
    ) -> None:
        """
        Process documents from the input directory asynchronously.
        
        Args:
            input_dir: Directory containing input documents
            save_chunks: Whether to save processed chunks
            save_embeddings: Whether to save generated embeddings
        """
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
                await self.graph_constructor.aconstruct_graph(documents)
            except Exception as e:
                self.logger.warning(f"Graph construction failed: {str(e)}")
            
        # Build retrieval index
        self.retrieval_system.build_index(documents)
        
        self.logger.info("Document processing completed successfully")
        
        # Cleanup raw documents after successful processing
        raw_docs_dir = os.path.join('data', 'raw_documents')
        if os.path.exists(raw_docs_dir):
            shutil.rmtree(raw_docs_dir)
            os.makedirs(raw_docs_dir)
            self.logger.info(f"Cleaned raw documents directory at {raw_docs_dir}")

    @log_errors(logging.getLogger(__name__))
    def query(
        self,
        query: str,
        mode: str = "Hybrid",
        top_k: int = 100,
        rerank_top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
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
        self.logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Determine actual search mode
        actual_mode = mode if self.graph_constructor is not None else "Dense"
        if actual_mode != mode:
            self.logger.warning(f"Falling back to {actual_mode} mode as graph is not available")
            
        # Perform search
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