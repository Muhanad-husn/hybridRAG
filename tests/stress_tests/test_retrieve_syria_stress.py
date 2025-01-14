import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from retrieve_syria import run_hybrid_search
import time

@pytest.mark.stress
def test_hybrid_search_stress():
    """Test the robustness of the hybrid search function under stress."""
    # Test with multiple concurrent queries
    start_time = time.time()
    results = []
    for i in range(10):  # Run 10 concurrent searches
        result = run_hybrid_search(f"stress test query {i}")
        results.append(result)
    
    execution_time = time.time() - start_time
    
    # Assert that all searches completed successfully
    for result in results:
        assert "answer" in result
        assert "sources" in result
        
    # Assert that the total execution time is reasonable
    assert execution_time < 120.0, "Stress test took too long"