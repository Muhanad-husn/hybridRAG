import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from retrieve_syria import run_hybrid_search
import time

@pytest.mark.performance
def test_hybrid_search_performance():
    """Test the performance of the hybrid search function."""
    # Test with a simple query
    start_time = time.time()
    result = run_hybrid_search("test query")
    execution_time = time.time() - start_time
    
    # Assert that the search completes within 15 seconds
    assert execution_time < 15.0, "Hybrid search took too long"
    
    # Assert that the result contains the expected keys
    assert "answer" in result
    assert "sources" in result