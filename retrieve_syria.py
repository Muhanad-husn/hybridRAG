import os
import sys
import re
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
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
from src.input_layer.translator import Translator

# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize translator lazily
_translator = None
def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator
def run_hybrid_search(query: str, original_lang: Optional[str] = None, original_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Run hybrid search with both dense retrieval and graph analysis,
    then process results with LLM to generate an answer.
    
    Args:
        query: The search query (in English)
        original_lang: Original language of query if translated (e.g., 'ar' for Arabic)
        original_query: Original query before translation if applicable
    
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

        # Store the raw input data used for LLM
        llm_input_data = {
            'context_parts': context_parts,
            'combined_context': context,
            'query': query,
            'original_query': original_query
        }
        
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
        
        # Get LLM response in English
        english_answer = llm_response.get("content", "")
        error = llm_response.get("error")
        
        # Always keep English answer
        answer = english_answer

        # Save English response to HTML
        if english_answer:
            try:
                from app import create_result_html
                logger.info("Saving English response to HTML...")
                english_filepath = create_result_html(
                    content=english_answer,
                    query=query,
                    translated_query="",
                    sources=sources,
                    is_arabic=False
                )
                logger.info(f"English response saved to: {english_filepath}")
            except Exception as e:
                logger.error(f"Error saving English response to HTML: {str(e)}")

        # Always translate to Arabic
        arabic_answer = None
        if english_answer:
            try:
                translator = get_translator()
                logger.info("Translating LLM response to Arabic...")
                arabic_answer = translator.translate(english_answer, source_lang='en', target_lang='ar')
                logger.info(f"Translation completed. Length: {len(arabic_answer)}")

                if not arabic_answer:
                    raise ValueError("Arabic translation is empty")

                # Save Arabic response to HTML
                try:
                    from app import create_result_html
                    logger.info("Saving Arabic response to HTML...")
                    arabic_filepath = create_result_html(
                        content=arabic_answer,
                        query=original_query or query,
                        translated_query=query if original_query else "",
                        sources=sources,
                        is_arabic=True
                    )
                    logger.info(f"Arabic response saved to: {arabic_filepath}")
                    
                    # Verify file was created
                    if not os.path.exists(arabic_filepath):
                        raise FileNotFoundError(f"Arabic HTML file not found at: {arabic_filepath}")
                    
                    logger.info(f"Verified Arabic HTML file exists at: {arabic_filepath}")
                except Exception as e:
                    logger.error(f"Error saving Arabic response to HTML: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

            except Exception as e:
                logger.error(f"Error in Arabic processing: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Don't fallback to English for Arabic answer
                arabic_answer = None
        
        # Calculate confidence score based on multiple factors
        confidence = 0
        if results:
            # Factor 1: Relevance scores from search results (30%)
            relevance_scores = [float(r.get('score', 0.0)) for r in results[:5]]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            relevance_confidence = min(avg_relevance * 30, 30)  # Max 30% from relevance

            # Factor 2: Number of sources (30%)
            source_count = len(sources)
            source_confidence = min(source_count * 6, 30)  # 6% per source, max 30%

            # Factor 3: Answer completeness (40%)
            answer_length = len(english_answer) if english_answer else 0
            length_confidence = min(answer_length / 1000 * 40, 40)  # Max 40% for 1000+ chars

            # Combine factors
            confidence = int(relevance_confidence + source_confidence + length_confidence)
            
            # Log confidence calculation
            logger.info(f"Confidence calculation:")
            logger.info(f"- Relevance confidence: {relevance_confidence:.1f}%")
            logger.info(f"- Source confidence: {source_confidence:.1f}%")
            logger.info(f"- Length confidence: {length_confidence:.1f}%")
            logger.info(f"- Total confidence: {confidence}%")

        # Extract vector data from results
        vector_data = []
        try:
            for result in results:
                if isinstance(result, dict):  # Ensure result is a dictionary
                    vector = result.get('vector')
                    if vector is not None:
                        # Handle different vector formats
                        if hasattr(vector, 'tolist'):
                            values = vector.tolist()
                        elif isinstance(vector, (list, tuple)):
                            values = list(vector)
                        else:
                            continue  # Skip if vector is in unexpected format
                            
                        vector_data.append({
                            'values': values,
                            'score': float(result.get('score', 0.0))
                        })
        except Exception as e:
            logger.error(f"Error extracting vector data: {str(e)}")

        # Extract graph relationships
        graph_data = []
        try:
            for result in results:
                if isinstance(result, dict) and result.get('meta') == 'graph_relationships':
                    content = result.get('content', '')
                    if isinstance(content, str):
                        relationships = content.split('\n')
                        for rel in relationships:
                            if ' -> ' in rel:
                                parts = rel.strip().split(' -> ')
                                if len(parts) == 3:
                                    graph_data.append({
                                        'subject': parts[0],
                                        'predicate': parts[1],
                                        'object': parts[2]
                                    })
        except Exception as e:
            logger.error(f"Error extracting graph data: {str(e)}")

        # Prepare base response
        response = {
            "query": query,
            "original_query": original_query or query,
            "answer": english_answer,
            "arabic_answer": None,  # Initialize as None
            "english_answer": english_answer,
            "error": error,
            "sources": sources,
            "language": original_lang or 'en',
            "english_file": None,
            "arabic_file": None,
            "confidence": confidence,
            "llm_input": {
                "context": context
            }
        }

        # Add English file if available
        try:
            if 'english_filepath' in locals() and os.path.exists(english_filepath):
                logger.info(f"Adding English file to response: {english_filepath}")
                response["english_file"] = os.path.basename(english_filepath)
        except Exception as e:
            logger.error(f"Error adding English file to response: {str(e)}")

        # Add Arabic content and file if available
        try:
            if arabic_answer and 'arabic_filepath' in locals() and os.path.exists(arabic_filepath):
                logger.info("Adding Arabic content and file to response")
                response["arabic_answer"] = arabic_answer
                response["arabic_file"] = os.path.basename(arabic_filepath)
                logger.info(f"Arabic content length: {len(arabic_answer)}")
                logger.info(f"Arabic file: {response['arabic_file']}")
        except Exception as e:
            logger.error(f"Error adding Arabic content to response: {str(e)}")

        # Log final response state
        logger.info("Final response state:", {
            "has_english": bool(response["answer"]),
            "has_arabic": bool(response["arabic_answer"]),
            "english_file": response["english_file"],
            "arabic_file": response["arabic_file"]
        })

        return response
            
            
    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        raise

def main():
    """Main entry point for retrieval."""
    try:
        if len(sys.argv) < 2:
            print("Please provide a query as a command line argument")
            sys.exit(1)
            
        # Get query from command line
        original_query = sys.argv[1]
        
        # Get translator instance
        translator = get_translator()
        
        # Detect language and translate if needed
        is_arabic = translator.is_arabic(original_query)
        
        if is_arabic:
            # Translate query to English
            english_query = translator.translate(original_query, source_lang='ar', target_lang='en')
            print(f"Translated query: {english_query}")
            result = run_hybrid_search(english_query, original_lang='ar', original_query=original_query)
        else:
            result = run_hybrid_search(original_query)
        
        if result.get("error"):
            print(f"Error from LLM: {result['error']}")
            return
            
        if not result["answer"]:
            print("Warning: No answer received from LLM")
            return
            
        # Get the answer
        answer = result["answer"].strip()
        
        # Format answer
        def format_answer(text: str) -> str:
            text = re.sub(r'^(.*?)\n', r'\1\n\n', text)  # Add space after title
            text = re.sub(r'([.!?])\n', r'\1\n\n', text)  # Add space between paragraphs
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove extra newlines
            text = re.sub(r'\n*Sources:', '\n\nSources:', text)  # Space before Sources
            return text
        
        # Print responses
        if result.get("language") == "ar":
            print("\nEnglish Response:")
            print("-" * 80)
            print(format_answer(result.get("english_answer", "")))
            print("-" * 80)
            print("\nArabic Response:")
            print("-" * 80)
            print(format_answer(answer))
        else:
            print(format_answer(answer))
        
        # Print sources
        print("\nSources:")
        for source in sorted(result["sources"]):
            print(f"- {source}")
            
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        raise

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()