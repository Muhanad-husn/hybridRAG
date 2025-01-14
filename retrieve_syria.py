import os
import sys
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
from src.utils.formatter import format_result

# Suppress all logging except errors
logging.getLogger().setLevel(logging.ERROR)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)

def run_hybrid_search(query: str):
    """Run hybrid search with both dense retrieval and graph analysis."""
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
            need_processing = not (os.path.exists(embeddings_dir) and os.listdir(embeddings_dir))
            
            # Load processed chunks if they exist
            chunks_dir = os.path.join("data", "processed_chunks")
            if os.path.exists(chunks_dir) and os.listdir(chunks_dir):
                documents = []
                for chunk_file in os.listdir(chunks_dir):
                    if chunk_file.endswith('.txt'):
                        with open(os.path.join(chunks_dir, chunk_file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            source = chunk_file.split('_', 2)[-1]  # Get original source from chunk filename
                            documents.append(Document(page_content=content, metadata={"source": source}))
                
                if need_processing:
                    # Generate and save embeddings
                    documents = embedding_generator.process_documents(documents)
                    embedding_generator.save_embeddings(documents)
                    retrieval_system.build_index(documents)
                else:
                    # Just initialize retrieval system with documents
                    retrieval_system.build_index(documents)
            else:
                # Process documents from scratch
                input_dir = os.path.join("data", "raw_documents")
                documents = document_processor.process_directory(input_dir)
                if not documents:
                    raise ValueError("No documents were processed successfully")
                
                # Save chunks
                document_processor.save_processed_chunks(documents, chunks_dir)
                
                # Generate embeddings
                documents = embedding_generator.process_documents(documents)
                embedding_generator.save_embeddings(documents)
                
                # Build retrieval index
                retrieval_system.build_index(documents)
        except Exception as e:
            print(f"Error preparing documents: {str(e)}")
            raise
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        try:
            # Get vector store results first
            vector_results = retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=None,  # Don't include graph results yet
                top_k=20,
                mode="Dense"  # Only dense retrieval
            )
            
            # Rerank vector results and keep exactly 10
            reranked_vector_results = retrieval_system.rerank_results(
                query=query,
                results=vector_results,
                top_k=10  # Keep exactly 10 after reranking
            )[:10]  # Ensure we have exactly 10
            
            # Get graph results
            graph_results = retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=graph_constructor,
                top_k=1,  # We only want the graph analysis
                mode="Hybrid"  # Get graph results
            )
            
            # Filter to keep only the graph analysis result
            graph_analysis = [r for r in graph_results if r.get('meta') == 'graph_relationships'][:1]  # Take only first graph result
            
            # Combine exactly 10 reranked vector results with 1 graph analysis
            combined_results = reranked_vector_results + graph_analysis
            
            # Sort by score (no need to deduplicate since we're controlling the counts)
            results = sorted(
                combined_results,
                key=lambda x: float(x.get('score', 0.0)),
                reverse=True
            )
            
        except Exception as e:
            print(f"Error in search pipeline: {str(e)}")
            raise
        
        # Print results in LLM-friendly format
        print(f"User Query: {query}\n")
        print("Retrieved Context (in order of relevance):")
        print("=" * 80)
        
        for result in results:
            formatted = format_result(result, format_type="text")
            print(formatted)
            
    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        raise

def main():
    """Main entry point for retrieval."""
    try:
        if len(sys.argv) < 2:
            print("Please provide a query as a command line argument")
            sys.exit(1)
        query = sys.argv[1]
        run_hybrid_search(query)
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        raise

if __name__ == "__main__":
    main()