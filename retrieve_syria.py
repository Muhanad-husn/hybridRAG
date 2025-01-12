import os
import sys
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.processing_layer.graph_constructor import GraphConstructor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def format_result(result):
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
            properties = properties.replace('{', '').replace('}', '').replace('"', '')
        property_text = ', '.join(f"{k}: {v}" for k, v in properties.items() if k != 'source')
        
        # Format edges
        edge_info = []
        for edge in edges:
            target_props = edge.get('properties', {})
            if isinstance(target_props, str):
                target_props = target_props.replace('{', '').replace('}', '').replace('"', '')
            edge_info.append(f"- {edge['type']} -> {edge['target']} ({target_props})")
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
            text = text.strip()[:500]
        meta = result.get('meta', 'unknown')
        score = result.get('score', 0.0)
        
        return (
            f"Content: {text}...\n"
            f"Source: {meta}\n"
            f"Relevance: {score:.4f}"
        )

def load_and_process_document(file_path):
    """Load and split document into chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    texts = text_splitter.split_text(content)
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": file_path}
        ) for chunk in texts
    ]
    
    return documents

def main():
    # Initialize components
    retrieval = HybridRetrieval()
    graph = GraphConstructor()
    
    # Define the query
    query = "What were the key factors that escalated the peaceful demonstrations in Syria into a full-scale armed conflict?"
    
    # Initialize retrieval system and load existing graph
    print("Loading document and initializing retrieval...")
    documents = load_and_process_document("data/raw_documents/test_document.txt")
    
    # Add documents to the vector store
    print("Building vector index...")
    retrieval.build_index(documents)
    
    # Ensure graph is loaded for hybrid search
    print(f"\nQuery: {query}\n")
    
    # Run searches once
    for mode in ["Dense", "Hybrid"]:
        print(f"\n{mode} Search Results:")
        results = retrieval.hybrid_search(
            query=query,
            query_embedding=None,
            graph=None if mode == "Dense" else graph,
            top_k=5,
            mode=mode
        )
        
        # Print results
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(format_result(result))
            print("-" * 80)
        
        print("=" * 100)

if __name__ == "__main__":
    main()