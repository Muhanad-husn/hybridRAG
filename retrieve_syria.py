import os
import sys
import re
import json
import logging
import traceback
from typing import Dict, Any, List
from contextlib import redirect_stderr
from io import StringIO
# Suppress all warnings and logging
import warnings
warnings.filterwarnings('ignore')
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
os.environ['FAISS_LOGGING_LEVEL'] = '0'  # Suppress FAISS logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('faiss').setLevel(logging.CRITICAL)

# Import after logging configuration
from langchain.schema import Document
from langchain.schema import Document
from src.utils.logger import setup_logger
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.utils.formatter import format_result
from src.tools.openrouter_client import OpenRouterClient

# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
def run_hybrid_search(query: str) -> Dict[str, Any]:
    """
    Run hybrid search with both dense retrieval and graph analysis,
    then process results with LLM to generate an answer.
    
    Returns:
        Dict containing retrieved context and LLM-generated answer
    """
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
        # Format results for LLM
        context_parts = []
        for result in results:
            formatted = format_result(result, format_type="text")
            context_parts.append(formatted)
        
        context = "\n\n".join(context_parts)
        
        # Extract sources using regex
        import re
        sources = set()
        source_pattern = r'\[([^(\]]+)'  # Match everything between [ and ( or ]
        for part in context_parts:
            matches = re.findall(source_pattern, part)
            for match in matches:
                source = match.strip()
                if source != 'graph_relationships':  # Exclude graph relationships
                    sources.add(source)
        sources = sorted(list(sources))
        
        # Initialize OpenRouter client
        llm_client = OpenRouterClient()
        
        # Create system prompt
        system_prompt = """You are a well-informed academic assistant. Your goal is to provide structured, educational, and accessible responses in a semi-academic tone. Specifically:

    Base your content on the provided context. If the context does not contain enough information, acknowledge this.
    Adopt an article-like structure with paragraphs:
        Introduction: Briefly set the stage.
        Body: Present ideas in paragraph form with smooth transitions. Use subheadings if needed, but rely on paragraphs rather than bullet points.
        Conclusion: Summarize the key points in a final paragraph.
    Avoid bullet points except for minor lists that must be itemized. When possible, integrate details into sentences rather than listing them.
    Semi-Academic Tone: Maintain clarity and accessibility for students or researchers in fields like sociopolitical, historical, or socioeconomic studies.
    Acknowledge Gaps: If certain details are missing, explicitly note these gaps."""

        # Create user prompt
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear and accurate answer based solely on the information provided in the context above."""
        
        # Get LLM response
        llm_response = llm_client.get_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,  # Higher temperature for more creative and natural academic tone
            max_tokens=1000
        )
        
        # Return answer and sources
        return {
            "query": query,
            "answer": llm_response.get("content", ""),
            "error": llm_response.get("error"),
            "sources": sources  # Include extracted sources
        }
            
            
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
        result = run_hybrid_search(query)
        
        if result.get("error"):
            print(f"Error from LLM: {result['error']}")
            return
            
        if not result["answer"]:
            print("Warning: No answer received from LLM")
            return
            
        # Format and print answer
        answer = result["answer"].strip()
        
        # Add space after title
        answer = re.sub(r'^(.*?)\n', r'\1\n\n', answer)
        
        # Add space between paragraphs
        answer = re.sub(r'([.!?])\n', r'\1\n\n', answer)
        
        # Remove any triple or more newlines
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Ensure proper spacing before Sources
        answer = re.sub(r'\n*Sources:', '\n\nSources:', answer)
        
        # Print formatted answer and sources
        print(answer)
        print("\nSources:")
        for source in sorted(result["sources"]):
            print(f"- {source}")
            
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        raise

if __name__ == "__main__":
    main()