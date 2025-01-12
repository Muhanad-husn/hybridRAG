import os
import sys
import logging
from typing import List, Dict, Any

# Add src directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.processing_layer.graph_constructor import GraphConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            f"Relevance: {node.get('relevance_score', 0.0):.4f}\n"
            f"Connections:\n{edge_summary}"
        )
    else:
        # Handle reranked results
        text = result.get('text', '')
        if isinstance(text, str):
            text = text.strip()[:500]  # Show more content, up to 500 chars
        meta = result.get('meta', 'unknown')
        score = result.get('score', 0.0)
        
        return (
            f"Content: {text}...\n"
            f"Source: {meta}\n"
            f"Relevance: {score:.4f}"
        )

def test_retrieval():
    """Test the retrieval functionality."""
    try:
        logger.info("Starting retrieval test")
        
        # Initialize components
        retrieval = HybridRetrieval()
        graph = GraphConstructor()
        
        # Test queries
        test_queries = [
            "What were the key factors that escalated the peaceful demonstrations in Syria into a full-scale armed conflict?",
            "How has the Syrian Civil War impacted the displacement of people both internally and internationally?",
            "What role has Russia played in shaping the dynamics of the Syrian Civil War?"
        ]
        
        # Process each query
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}:")
            print(query)
            print("\nResults:")
            
            # Test different retrieval modes
            for mode in ["Dense", "Hybrid"]:
                print(f"\n{mode} Search Results:")
                results = retrieval.hybrid_search(
                    query=query,
                    query_embedding=None,  # Not used directly
                    graph=graph if mode == "Hybrid" else None,
                    top_k=5,  # Limit to top 5 most relevant results
                    mode=mode
                )
                
                # Print results
                for idx, result in enumerate(results, 1):
                    print(f"\nResult {idx}:")
                    print(format_result(result))
                    print("-" * 80)
                
                print("=" * 100)
        
        logger.info("Retrieval test completed")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_retrieval()