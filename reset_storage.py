import os
import sys
import yaml
from src.utils.logger import setup_logger, get_logger
from src.utils.cache_handler import DocumentCache
from Process_files import HyperRAG

# Initialize logger
setup_logger()
logger = get_logger(__name__)

def reset_storage():
    try:
        # Initialize HyperRAG
        rag_system = HyperRAG(logger=logger)
        
        # Reset storage
        rag_system.reset_storage()
        logger.info("Storage reset completed")
        
        # Clear cache
        cache = DocumentCache()
        cache.clear()
        logger.info("Cache cleared")
        
        # Get vector store stats
        vector_count = len(os.listdir(os.path.join("data", "embeddings"))) - 1  # Subtract 1 for index.faiss
        
        # Get graph stats
        nodes_file = os.path.join("data", "graphs", "nodes.csv")
        edges_file = os.path.join("data", "graphs", "edges.csv")
        
        node_count = 0
        edge_count = 0
        
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r') as f:
                node_count = sum(1 for line in f) - 1  # Subtract 1 for header
                
        if os.path.exists(edges_file):
            with open(edges_file, 'r') as f:
                edge_count = sum(1 for line in f) - 1  # Subtract 1 for header
        
        print(f"Storage reset and cache clearing completed successfully.")
        print(f"Vector count: {vector_count}")
        print(f"Node count: {node_count}")
        print(f"Edge count: {edge_count}")
        
    except Exception as e:
        logger.error(f"Error resetting storage and clearing cache: {str(e)}")
        print(f"An error occurred while resetting storage and clearing cache: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    reset_storage()