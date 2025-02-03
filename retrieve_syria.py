import os
import sys
import re
import yaml
import logging
import traceback
import tiktoken
from typing import Dict, Any, Optional
from datetime import datetime
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
os.environ['FAISS_LOGGING_LEVEL'] = '0'  # Suppress FAISS logging

# Import dependencies
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.utils.formatter import format_result
from src.tools.openrouter_client import OpenRouterClient
from src.input_layer.translator import Translator
from src.utils.config_handler import config

# Initialize logger
def initialize_logger(external_logger=None):
    global logger
    logger = external_logger or logging.getLogger(__name__)

# Initialize logger with default
initialize_logger()

# Initialize translator lazily
_translator = None
def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

# Singleton instance of HybridRetrieval
_hybrid_retrieval_instance = None

def get_hybrid_retrieval_instance(config_path: str = "config/config.yaml") -> HybridRetrieval:
    global _hybrid_retrieval_instance
    if _hybrid_retrieval_instance is None:
        _hybrid_retrieval_instance = HybridRetrieval(config_path)
    return _hybrid_retrieval_instance

def run_hybrid_search(query: str, external_logger: Optional[logging.Logger] = None, original_lang: Optional[str] = None,
                     original_query: Optional[str] = None, translate: bool = True, rerank_count: int = 15,
                     max_tokens: int = 3000, temperature: float = 0.0, context_length: int = 16000) -> Dict[str, Any]:
    if external_logger:
        initialize_logger(external_logger)
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
    initialize_logger()
    logger.info(f"Starting run_hybrid_search with query: {query}, translate: {translate}")
    try:
        # Load config
        config_path = "config/config.yaml"
        config.load_config(config_path)

        # Load top_k from config
        top_k = config.get('retrieval.top_k', 100)  # Default to 100 if not found

        # Initialize components
        document_processor = DocumentProcessor()  # Updated to use centralized config
        embedding_generator = EmbeddingGenerator(config_path)
        graph_constructor = GraphConstructor(config_path)
        retrieval_system = get_hybrid_retrieval_instance(config_path)
        
        try:
            # Verify vector store exists
            if not (os.path.exists(embedding_generator.embeddings_dir) and
                   os.listdir(embedding_generator.embeddings_dir)):
                initialize_logger()
                logger.error("Vector store not found. Please process documents using HyperRAG to initialize the system.")
                raise ValueError("Vector store not initialized. Please process documents using HyperRAG to initialize the system.")
                    
            initialize_logger()
            logger.info("Using existing embeddings")
        except Exception as e:
            initialize_logger()
            logger.error(f"Error preparing documents: {str(e)}")
            raise
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Initialize variables for the loop
        current_rerank_count = rerank_count
        context_tokens = float('inf')
        available_tokens = context_length - max_tokens
        enc = tiktoken.encoding_for_model("gpt-4")  # Using GPT-4 encoding as a standard
    
        while context_tokens > available_tokens and current_rerank_count >= 5:
            try:
                # Get vector store results
                initialize_logger()
                logger.info(f"Retrieving dense vector results with rerank_count: {current_rerank_count}...")
                vector_results = retrieval_system.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    graph=None,  # Don't include graph results yet
                    top_k=top_k,   # Use the value from config
                    mode="Dense" # Only dense retrieval
                )
                
                if not vector_results:
                    initialize_logger()
                    logger.warning("No vector results found")
                    return {
                        "query": query,
                        "answer": "I apologize, but I couldn't find any relevant information in the vector store.",
                        "error": "No vector results found",
                        "sources": [],
                        "confidence": 0
                    }
                
                initialize_logger()
                logger.info(f"Found {len(vector_results)} initial vector results")
                
                # Rerank vector results
                initialize_logger()
                logger.info("Reranking vector results...")
                reranked_vector_results = retrieval_system.rerank_results(
                    query=query,
                    results=vector_results,
                    top_k=current_rerank_count
                )
                
                if not reranked_vector_results:
                    initialize_logger()
                    logger.warning("No results after reranking")
                    return {
                        "query": query,
                        "answer": "I apologize, but the reranking process didn't yield any relevant results.",
                        "error": "No results after reranking",
                        "sources": [],
                        "confidence": 0
                    }
                
                initialize_logger()
                logger.info(f"Reranked to {len(reranked_vector_results)} results")
                
                # Get graph results - keep graph results proportional to rerank_count
                min_graph_results = 3  # Minimum number of graph results to include
                graph_k = max(min_graph_results, min(current_rerank_count // 3, 10))  # Scale graph results with rerank_count
                initialize_logger()
                logger.info(f"Retrieving graph results with graph_k: {graph_k}")
                graph_results = retrieval_system.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    graph=graph_constructor,
                    top_k=graph_k,
                    mode="Hybrid"  # Get graph results
                )
                
                # Include all graph results without filtering
                graph_analysis = graph_results
                initialize_logger()
                logger.info(f"Including {len(graph_analysis)} graph results")
                
                # Ensure we have at least min_graph_results
                graph_analysis = graph_analysis[:max(min_graph_results, len(graph_analysis))]
                
                # Adjust current_rerank_count to make room for graph results
                current_rerank_count = max(current_rerank_count - len(graph_analysis), 0)
                
                # Combine graph and vector results
                combined_results = graph_analysis + reranked_vector_results[:current_rerank_count]
                initialize_logger()
                logger.info(f"Added {len(graph_analysis)} graph results and {len(combined_results) - len(graph_analysis)} vector results to combined results")
                
                # Sort by score
                results = sorted(
                    combined_results,
                    key=lambda x: float(x.get('score', 0.0)),
                    reverse=True
                )
                
                # Format results for LLM
                context_parts = []
                
                # First add dense retrieval results
                for result in reranked_vector_results:
                    if isinstance(result, dict) and 'text' in result:
                        formatted = format_result(result, format_type="text")
                        if formatted.strip():  # Only add non-empty results
                            context_parts.append(formatted)
                            
                # Then add graph analysis if available
                if graph_analysis:
                    for result in graph_analysis:
                        formatted = format_result(result, format_type="text")
                        if formatted.strip():
                            context_parts.append(formatted)
                
                # Join all parts with double newlines
                context = "\n\n".join(context_parts)
                
                # Check context length
                context_tokens = len(enc.encode(context))
                initialize_logger()
                logger.info(f"Context length: {context_tokens} tokens")
                
                if context_tokens <= available_tokens:
                    break
                
                # Reduce rerank_count for next iteration
                current_rerank_count = max(5, current_rerank_count - 5)
                initialize_logger()
                logger.warning(f"Context length ({context_tokens} tokens) exceeds available tokens ({available_tokens}). "
                               f"Reducing rerank_count to {current_rerank_count}")
                
            except Exception as e:
                initialize_logger()
                logger.error(f"Error in search pipeline: {str(e)}")
                raise
    
        if context_tokens > available_tokens:
            initialize_logger()
            logger.warning(f"Could not reduce context length below the limit. Final context length: {context_tokens} tokens")
    
        if not context.strip():
            initialize_logger()
            logger.warning("No context generated from search results")
            return {
                "query": query,
                "answer": "I apologize, but I couldn't find any relevant information to answer your question.",
                "error": "No relevant context found",
                "sources": [],
                "confidence": 0
            }

        # Extract unique document sources
        sources = set()
        source_pattern = r'\[(.*?)\s*\(Relevance:'  # Match filename before (Relevance:
        for part in context_parts:
            matches = re.findall(source_pattern, part)
            for match in matches:
                source = match.strip()
                if source and source != 'graph_relationships':
                    source = os.path.splitext(source)[0]  # Remove file extension
                    sources.add(source)
        sources = sorted(list(sources))
        
        # Initialize OpenRouter client with answer model from config
        llm_client = OpenRouterClient(model=config.get('llm.answer_model'))
        initialize_logger()
        logger.info(f"Using answer model: {config.get('llm.answer_model')}")
        
        # Create system prompt
        system_prompt = """You are a well-informed academic assistant. Your goal is to provide structured, educational, and accessible responses in a semi-academic tone. Specifically:
    
            Start your response with a concise, relevant title on the first line, without prefixing it with 'Title:'.
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
    
    Please provide a clear and accurate answer based solely on the information provided in the context above. Begin your response with a concise, relevant title on the first line, without prefixing it with 'Title:'."""
        
        # Get LLM response
        llm_response = llm_client.get_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Get LLM response in English
        english_answer = llm_response.get("content", "")
        error = llm_response.get("error")
        
        # Always keep English answer
        answer = english_answer
    
        # Only translate to Arabic if translation is enabled
        arabic_answer = None
        if english_answer and translate:
            try:
                initialize_logger()
                logger.info("Attempting to translate LLM response to Arabic")
                translator = get_translator()
                arabic_answer = translator.translate(english_answer, source_lang='en', target_lang='ar')
                initialize_logger()
                logger.info(f"Arabic translation completed. Length: {len(arabic_answer)}")
    
                if not arabic_answer:
                    raise ValueError("Arabic translation is empty")
    
            except Exception as e:
                initialize_logger()
                logger.error(f"Error in Arabic processing: {str(e)}")
                initialize_logger()
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Don't fallback to English for Arabic answer
                arabic_answer = None
        elif english_answer and not translate:
            initialize_logger()
            logger.info("Skipping Arabic translation as per user preference")
        
        # Calculate UI-optimized confidence score with dynamic scaling
        confidence = 0
        if results:
            # Get base probabilities (0-1 range)
            # Scale number of scores to consider based on rerank_count
            scores_to_consider = max(5, min(rerank_count // 3, 15))
            relevance_scores = [float(r.get('score', 0.0)) for r in results[:scores_to_consider]]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            relevance_prob = avg_relevance
            
            # Scale source count expectation with rerank_count
            expected_sources = max(5, min(rerank_count // 4, 20))
            source_count = len(sources)
            source_prob = min(source_count / expected_sources, 1.0)
            
            # Scale expected answer length with rerank_count
            expected_length = min(1000 * (rerank_count / 15), 3000)  # Scale up to 3000 chars max
            answer_length = len(english_answer) if english_answer else 0
            length_prob = min(answer_length / expected_length, 1.0)
            
            # Calculate raw multiplicative confidence
            raw_confidence = relevance_prob * source_prob * length_prob
            
            # Scale confidence for UI display (65-95 range)
            MIN_CONFIDENCE = 65
            MAX_CONFIDENCE = 95
            RANGE = MAX_CONFIDENCE - MIN_CONFIDENCE
            confidence = int(MIN_CONFIDENCE + (raw_confidence * RANGE))
            
            # Log detailed calculation with dynamic scaling info
            initialize_logger()
            logger.info(f"Dynamic confidence calculation:")
            initialize_logger()
            logger.info(f"- Using {scores_to_consider} scores for relevance calculation")
            initialize_logger()
            logger.info(f"- Expected sources scaled to {expected_sources}")
            initialize_logger()
            logger.info(f"- Expected length scaled to {expected_length} characters")
            
            # Log detailed calculation
            initialize_logger()
            logger.info(f"Confidence calculation:")
            initialize_logger()
            logger.info(f"- Relevance score: {relevance_prob:.3f} ({relevance_prob*100:.1f}%)")
            initialize_logger()
            logger.info(f"- Source coverage: {source_prob:.3f} ({source_prob*100:.1f}%)")
            initialize_logger()
            logger.info(f"- Answer completeness: {length_prob:.3f} ({length_prob*100:.1f}%)")
            initialize_logger()
            logger.info(f"- Raw confidence: {raw_confidence:.3f} ({raw_confidence*100:.1f}%)")
            initialize_logger()
            logger.info(f"- UI-scaled confidence: {confidence}%")
    
        # Prepare base response
        initialize_logger()
        logger.info("Creating response dictionary")
        response = {
            "query": query,
            "original_query": original_query or query,
            "answer": english_answer,
            "arabic_answer": arabic_answer,
            "english_answer": english_answer,
            "error": error,
            "sources": sources,
            "language": original_lang or 'en',
            "confidence": confidence,
            "llm_input": {
                "context": context,
                "context_tokens": context_tokens if 'context_tokens' in locals() else None,
            },
            "raw_data": {
                "reranked_vector_results": reranked_vector_results,
                "graph_analysis": graph_analysis
            },
            "warning": f"Context length exceeded available tokens. Results automatically reduced from {rerank_count} to {current_rerank_count} for better processing."
            if current_rerank_count < rerank_count else None
        }
    
        initialize_logger()
        logger.info("Completed run_hybrid_search")
        return response
            
    except Exception as e:
        initialize_logger()
        logger.error(f"Error in hybrid search: {str(e)}")
        raise

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    # Main function implementation here if needed