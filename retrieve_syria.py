import os
import sys
import re
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from contextlib import redirect_stderr
from io import StringIO
from datetime import datetime
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
os.environ['FAISS_LOGGING_LEVEL'] = '0'  # Suppress FAISS logging

# Import dependencies
from langchain.schema import Document
from src.input_layer.document_processor import DocumentProcessor
from src.processing_layer.embedding_generator import EmbeddingGenerator
from src.processing_layer.graph_constructor import GraphConstructor
from src.retrieval_layer.hybrid_retrieval import HybridRetrieval
from src.utils.formatter import format_result
from src.tools.openrouter_client import OpenRouterClient
from src.input_layer.translator import Translator

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
            # Verify vector store exists
            if not (os.path.exists(embedding_generator.embeddings_dir) and
                   os.listdir(embedding_generator.embeddings_dir)):
                logger.error("Vector store not found. Please run src/main.py first to initialize the system.")
                raise ValueError("Vector store not initialized. Run src/main.py first.")
                
            logger.info("Using existing embeddings")
        except Exception as e:
            print(f"Error preparing documents: {str(e)}")
            raise
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        try:
            # Get vector store results first
            logger.info("Retrieving dense vector results...")
            vector_results = retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=None,  # Don't include graph results yet
                top_k=100,   # Get more results for better reranking
                mode="Dense" # Only dense retrieval
            )
            
            if not vector_results:
                logger.warning("No vector results found")
                return {
                    "query": query,
                    "answer": "I apologize, but I couldn't find any relevant information in the vector store.",
                    "error": "No vector results found",
                    "sources": [],
                    "confidence": 0
                }
            
            logger.info(f"Found {len(vector_results)} initial vector results")
            
            # Rerank vector results
            logger.info("Reranking vector results...")
            reranked_vector_results = retrieval_system.rerank_results(
                query=query,
                results=vector_results,
                top_k=15  # Keep top 15 after reranking
            )
            
            if not reranked_vector_results:
                logger.warning("No results after reranking")
                return {
                    "query": query,
                    "answer": "I apologize, but the reranking process didn't yield any relevant results.",
                    "error": "No results after reranking",
                    "sources": [],
                    "confidence": 0
                }
            
            logger.info(f"Reranked to {len(reranked_vector_results)} results")
            
            # Get graph results
            graph_results = retrieval_system.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                graph=graph_constructor,
                top_k=5,     # Allow for multiple graph relationships
                mode="Hybrid"  # Get graph results
            )
            
            # Filter to keep only the graph analysis results
            graph_analysis = [r for r in graph_results if r.get('meta') == 'graph_relationships']
            
            # Calculate how many graph results we can include while respecting the total limit
            remaining_slots = 15 - len(reranked_vector_results)  # User's configured limit minus vector results
            if remaining_slots > 0 and graph_analysis:
                # Add graph results up to the remaining limit
                graph_analysis = graph_analysis[:remaining_slots]
                combined_results = reranked_vector_results + graph_analysis
            else:
                combined_results = reranked_vector_results
            
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
        
        if not context.strip():
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

            except Exception as e:
                logger.error(f"Error in Arabic processing: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Don't fallback to English for Arabic answer
                arabic_answer = None
        
        # Calculate UI-optimized confidence score
        confidence = 0
        if results:
            # Get base probabilities (0-1 range)
            relevance_scores = [float(r.get('score', 0.0)) for r in results[:5]]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            relevance_prob = avg_relevance
            
            source_count = len(sources)
            source_prob = min(source_count / 5, 1.0)  # Normalized by expected sources (5)
            
            answer_length = len(english_answer) if english_answer else 0
            length_prob = min(answer_length / 1000, 1.0)  # Normalized by expected length (1000)
            
            # Calculate raw multiplicative confidence
            raw_confidence = relevance_prob * source_prob * length_prob
            
            # Scale confidence for UI display (65-95 range)
            MIN_CONFIDENCE = 65
            MAX_CONFIDENCE = 95
            RANGE = MAX_CONFIDENCE - MIN_CONFIDENCE
            confidence = int(MIN_CONFIDENCE + (raw_confidence * RANGE))
            
            # Log detailed calculation
            logger.info(f"Confidence calculation:")
            logger.info(f"- Relevance score: {relevance_prob:.3f} ({relevance_prob*100:.1f}%)")
            logger.info(f"- Source coverage: {source_prob:.3f} ({source_prob*100:.1f}%)")
            logger.info(f"- Answer completeness: {length_prob:.3f} ({length_prob*100:.1f}%)")
            logger.info(f"- Raw confidence: {raw_confidence:.3f} ({raw_confidence*100:.1f}%)")
            logger.info(f"- UI-scaled confidence: {confidence}%")

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

        # Generate HTML content
        english_result = None
        arabic_result = None

        def create_html_result(content, query, translated_query, sources, is_arabic):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            safe_title = ''.join(c for c in content.split('\n')[0] if c.isalnum() or c in (' ', '-', '_'))[:50]
            filename = f"{safe_title}.html"
            return {
                'html': f"""<html>
<body>
<h1>{content.split('\n')[0]}</h1>
{content}
<hr>
<p>Query: {query}</p>
<p>Sources: {', '.join(sources)}</p>
<p>Generated: {timestamp}</p>
</body>
</html>""",
                'filename': filename
            }

        if english_answer:
            english_result = create_html_result(
                content=english_answer,
                query=query,
                translated_query="",
                sources=sources,
                is_arabic=False
            )

        if arabic_answer:
            arabic_result = create_html_result(
                content=arabic_answer,
                query=original_query or query,
                translated_query=query if original_query else "",
                sources=sources,
                is_arabic=True
            )

        # Prepare base response
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
                "context": context
            }
        }

        # Add HTML content if available
        if english_result:
            response["english_html"] = english_result["html"]
            response["english_filename"] = english_result["filename"]

        if arabic_result:
            response["arabic_html"] = arabic_result["html"]
            response["arabic_filename"] = arabic_result["filename"]

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