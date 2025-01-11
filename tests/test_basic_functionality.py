import os
import sys
import pytest
import logging
from pathlib import Path

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / "src")
sys.path.append(src_dir)

from main import HyperRAG

# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def rag_system():
    """Fixture to create a HyperRAG instance."""
    return HyperRAG()

@pytest.fixture
def sample_documents(tmp_path):
    """Fixture to create sample test documents."""
    # Create a temporary directory for test documents
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()
    
    # Create a sample text document
    text_doc = docs_dir / "sample.txt"
    text_doc.write_text("""
    This is a sample document for testing the Hyper RAG system.
    It contains multiple sentences about different topics.
    The system should be able to process this document and extract meaningful information.
    We can test various features like document processing, embedding generation, and retrieval.
    """)
    
    return str(docs_dir)

def test_document_processing(rag_system, sample_documents):
    """Test basic document processing functionality."""
    try:
        # Process the sample documents
        rag_system.process_documents(
            input_dir=sample_documents,
            save_chunks=True,
            save_embeddings=True
        )
        
        # Verify that the processed files exist
        assert os.path.exists("data/processed_chunks")
        assert os.path.exists("data/embeddings")
        
        logger.info("Document processing test passed")
        
    except Exception as e:
        logger.error(f"Document processing test failed: {str(e)}")
        raise

def test_query_execution(rag_system, sample_documents):
    """Test query execution functionality."""
    try:
        # First process documents
        rag_system.process_documents(
            input_dir=sample_documents,
            save_chunks=True,
            save_embeddings=True
        )
        
        # Test queries
        test_queries = [
            "What topics are discussed?",
            "What is the system capable of?",
            "What features can be tested?"
        ]
        
        for query in test_queries:
            # Execute query
            results = rag_system.query(
                query=query,
                mode="Hybrid",
                top_k=5
            )
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Check result structure
            for result in results:
                assert 'text' in result
                assert 'meta' in result
                assert 'score' in result
                assert isinstance(result['score'], float)
                
            logger.info(f"Query test passed for: {query}")
        
        logger.info("All query execution tests passed")
        
    except Exception as e:
        logger.error(f"Query execution test failed: {str(e)}")
        raise

def test_graph_operations(rag_system, sample_documents):
    """Test knowledge graph operations."""
    try:
        # Process documents to build graph
        rag_system.process_documents(
            input_dir=sample_documents,
            save_chunks=True,
            save_embeddings=True
        )
        
        # Test graph clearing
        rag_system.clear_graph()
        
        # Rebuild graph
        rag_system.process_documents(
            input_dir=sample_documents,
            save_chunks=False,
            save_embeddings=False
        )
        
        logger.info("Graph operations test passed")
        
    except Exception as e:
        logger.error(f"Graph operations test failed: {str(e)}")
        raise

def test_error_handling(rag_system):
    """Test error handling for invalid inputs."""
    try:
        # Test with non-existent directory
        with pytest.raises(Exception):
            rag_system.process_documents(
                input_dir="non_existent_directory",
                save_chunks=True,
                save_embeddings=True
            )
        
        # Test with empty query
        with pytest.raises(Exception):
            rag_system.query(
                query="",
                mode="Invalid",
                top_k=5
            )
        
        logger.info("Error handling test passed")
        
    except Exception as e:
        logger.error(f"Error handling test failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])