import os
import json
import logging
import traceback
from typing import Dict, Any
from langchain.schema import Document
from src.utils.logger import setup_logger
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval

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
            f"Connections:\n{edge_summary}"
        )
    else:
        # Handle reranked results
        text = result.get('text', '')
        if isinstance(text, str):
            text = text.strip()[:500]  # Show more content, up to 500 chars
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

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_hybrid_search():
    """Run hybrid search with both dense retrieval and graph analysis."""
    logger.info("Starting retrieval process...")
    logger.info("Initializing retrieval components...")
    
    try:
        # Initialize components
        config_path = "config/config.yaml"
        document_processor = DocumentProcessor(config_path)
        embedding_generator = EmbeddingGenerator(config_path)
        graph_constructor = GraphConstructor(config_path)
        retrieval_system = HybridRetrieval(config_path)
        
        try:
            # Check if vector store exists
            embeddings_dir = os.path.join("data", "embeddings")
            logger.info(f"Checking vector store at {embeddings_dir}")
            
            if not os.path.exists(embeddings_dir):
                logger.info("Vector store directory not found")
                need_processing = True
            elif not os.listdir(embeddings_dir):
                logger.info("Vector store directory is empty")
                need_processing = True
            else:
                logger.info(f"Found existing vector store with files: {os.listdir(embeddings_dir)}")
                need_processing = False
            
            # Always process documents to get the Document objects
            input_dir = os.path.join("data", "raw_documents")
            logger.info(f"Processing documents from {input_dir}")
            documents = document_processor.process_directory(input_dir)
            if not documents:
                raise ValueError("No documents were processed successfully")
            
            if need_processing:
                logger.info("Vector store not found. Processing documents...")
                
                # Save chunks
                output_dir = os.path.join("data", "processed_chunks")
                document_processor.save_processed_chunks(documents, output_dir)
                
                # Generate embeddings
                documents = embedding_generator.process_documents(documents)
                
                # Save embeddings
                embedding_generator.save_embeddings(documents)
                
                # Build retrieval index
                logger.info("Building retrieval index...")
                retrieval_system.build_index(documents)
                logger.info("Retrieval index built successfully")
            else:
                logger.info("Using existing vector store...")
                # Add documents to retrieval system without regenerating embeddings
                retrieval_system.build_index(documents)
        except Exception as e:
            logger.error(f"Error checking/building vector store: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        logger.info("Successfully loaded vector store and graph")
        logger.info(f"Graph has {graph_constructor.graph.number_of_nodes()} nodes and {graph_constructor.graph.number_of_edges()} edges")
        
        # Example query
        query = "How has the Syrian Civil War impacted the displacement of people both internally and internationally?"
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        logger.info("Generated query embedding")
        
        # Perform hybrid search
        print("\nHybrid Search Results:")
        print("=" * 100)
        
        try:
            # Get initial hybrid search results
            logger.info("Getting initial hybrid search results...")
            initial_results = retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=graph_constructor,
                top_k=20,  # Get more results initially
                mode="Hybrid"  # Explicitly set hybrid mode
            )
            logger.info(f"Got {len(initial_results)} initial results")
            logger.debug(f"Initial results: {initial_results}")
            
            # Deduplicate results
            logger.info("Deduplicating results...")
            unique_results = retrieval_system.deduplicate_results(initial_results)
            logger.info(f"Got {len(unique_results)} unique results after deduplication")
            logger.debug(f"Unique results: {unique_results}")
            
            # Rerank results
            logger.info("Reranking results...")
            reranked_results = retrieval_system.rerank_results(
                query=query,
                results=unique_results,
                top_k=10  # Keep top 10 after reranking
            )
            logger.info(f"Got {len(reranked_results)} results after reranking")
            logger.debug(f"Reranked results: {reranked_results}")
            
            # Sort by score
            results = sorted(
                reranked_results,
                key=lambda x: float(x.get('score', 0.0)),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error in search pipeline: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        logger.info(f"Got {len(results)} results")
        
        # Print results
        for idx, result in enumerate(results, 1):
            logger.info(f"Processing result {idx}")
            print(f"\nResult {idx}:")
            formatted = format_result(result)
            print(formatted)
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise

def main():
    """Main entry point for retrieval."""
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting main process...")
        run_hybrid_search()
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        logger.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
        raise

if __name__ == "__main__":
    main()