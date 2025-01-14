import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from retrieve_syria import run_hybrid_search

@pytest.mark.integration
def test_hybrid_search_integration():
    """Test the integration of all modules in the hybrid search pipeline."""
    # Test with a simple query
    result = run_hybrid_search("test query")
    
    # Assert that the result contains the expected keys
    assert "answer" in result
    assert "sources" in result
    
    # Assert that the answer is not empty
    assert result["answer"].strip() != ""
    
    # Assert that sources is a list
    assert isinstance(result["sources"], list)