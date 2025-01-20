from flask import Flask, render_template, request, jsonify, send_file
import sys
import logging
import io
import os
from datetime import datetime
from collections import deque
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

# Initialize logger using the unified logging setup
setup_logger()
logger = get_logger(__name__)

# Initialize translator and search history
translator = Translator()
search_history = []  # List to store unique search queries
saved_results = []  # List to store saved result filenames

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Add request info to logger only for main endpoints
        if request.endpoint in ['search', 'save_result', 'generate_result']:
            logger.info(f"Request received: {request.method} {request.path}")
        return f(*args, **kwargs)
    return decorated_function

_font_verified = False
def verify_font():
    global _font_verified
    if _font_verified:
        return
        
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', 'fonts', 'NotoNaskhArabic-Regular.ttf')
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found at {font_path}")
        _font_verified = True
    except Exception as e:
        logger.error(f"Font verification failed: {str(e)}")
        raise

# Verify font once at startup
verify_font()

def create_result_html(content, query, translated_query, sources, is_arabic=False):
    try:
        # Create timestamp for display
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format content and extract title
        def format_content(text):
            # Replace HTML entities with actual characters
            text = text.replace('&#39;', "'")
            text = text.replace('&quot;', '"')
            
            # Split into paragraphs and filter out empty ones
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            # Extract first line as title
            title = paragraphs[0] if paragraphs else "Untitled"
            
            # Join with double newlines for proper paragraph spacing
            return title, '\n\n'.join(paragraphs)
        
        # Format content and get title
        title, formatted_content = format_content(content)
        
        # Clean the title for filename
        safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
        filename = f"{safe_title}.html"
        
        # Render HTML template
        html = render_template(
            'result_template.html',
            content=formatted_content,
            query=query,
            translated_query=translated_query,
            sources=sources,
            is_arabic=is_arabic,
            timestamp=timestamp
        )
        
        return {'html': html, 'filename': filename}
        
    except Exception as e:
        logger.error(f"Error generating HTML: {str(e)}")
        raise

@app.route('/')
@log_request
def home():
    return render_template('index.html')

@app.route('/search-history', methods=['GET'])
@log_request
def get_search_history():
    return jsonify({'history': search_history})

@app.route('/saved-results', methods=['GET'])
@log_request
def get_saved_results():
    return jsonify({'results': saved_results})

@app.route('/save-result', methods=['POST'])
@log_request
def save_result():
    try:
        data = request.get_json()
        content = data.get('content')
        filename = data.get('filename')
        html = data.get('html')
        
        if not all([content, filename, html]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), 'docs')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
            
        # Add to saved results (maintain uniqueness and limit to 25)
        if filename not in saved_results:
            if len(saved_results) >= 25:
                saved_results.pop()  # Remove oldest entry
            saved_results.insert(0, filename)  # Add new entry at the beginning
            
        logger.info(f"Result saved: {filename}")
        return jsonify({
            'message': 'Result saved successfully',
            'saved_results': saved_results,
            'filepath': filepath
        })
    except Exception as e:
        logger.error(f"Error saving result: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
@log_request
def get_logs():
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            # Get last 50 lines
            lines = deque(maxlen=50)
            for line in f:
                if 'INFO' in line and any(x in line for x in [
                    'Processing',
                    'Translating',
                    'Translation',
                    'Saving',
                    'response saved',
                    'Verified',
                    'Adding',
                    'Confidence',
                    'Request received'
                ]):
                    # Extract timestamp and message
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        timestamp = parts[0].strip()
                        message = parts[2].strip()
                        lines.append(f"[{timestamp}] {message}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_logs = []
            for log in lines:
                msg = log.split('] ', 1)[1] if '] ' in log else log
                if msg not in seen:
                    seen.add(msg)
                    unique_logs.append(log)
            
            return jsonify({'logs': unique_logs})
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@log_request
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        translate_enabled = data.get('translate', True)  # Default to True for backward compatibility
        
        if not query:
            logger.warning("Empty search query received")
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Update search history (maintain uniqueness and limit to 25)
        if query not in search_history:
            if len(search_history) >= 25:
                search_history.pop()  # Remove oldest entry
            search_history.insert(0, query)  # Add new query at the beginning

        # Detect if query is Arabic
        is_arabic = translator.is_arabic(query)
        
        # Get rerank count from request, default to 15 if not provided or invalid
        rerank_count = max(min(int(data.get('rerank_count', 15)), 80), 5)
        logger.info(f"Using rerank count: {rerank_count}")

        if is_arabic:
            logger.info(f"Processing Arabic query: {query}")
            # Translate query to English for internal processing only
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            # Pass original query without translation and respect translation preference
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query,
                                     translate=translate_enabled, rerank_count=rerank_count)
        else:
            logger.info(f"Processing English query: {query}")
            result = run_hybrid_search(query, translate=translate_enabled, rerank_count=rerank_count)

        # Generate HTML content without saving
        if result.get('answer'):
            english_result = create_result_html(
                content=result['answer'],
                query=query,
                translated_query="",
                sources=result.get('sources', []),
                is_arabic=False
            )
            result['english_html'] = english_result['html']
            result['english_filename'] = english_result['filename']

        # Only include Arabic translation if translation is enabled
        if translate_enabled and result.get('arabic_answer'):
            arabic_result = create_result_html(
                content=result['arabic_answer'],
                query=result.get('original_query', query),
                translated_query=query if result.get('original_query') else "",
                sources=result.get('sources', []),
                is_arabic=True
            )
            result['arabic_html'] = arabic_result['html']
            result['arabic_filename'] = arabic_result['filename']
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
@log_request
def serve_result(filename):
    try:
        docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
        filepath = os.path.join(docs_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({'error': 'File not found'}), 404
            
        try:
            response = send_file(
                filepath,
                mimetype='text/html',
                as_attachment=True,
                download_name=filename
            )
            return response
        except Exception as serve_error:
            raise serve_error
            
    except Exception as e:
        logger.error(f"Error serving result file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-result', methods=['POST'])
@log_request
def generate_result():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        content = data.get('content', '').strip()
        if not content:
            return jsonify({'error': 'Content cannot be empty'}), 400

        query = data.get('query', '').strip()
        translated_query = data.get('translatedQuery', '').strip()
        sources = [s for s in data.get('sources', []) if s.strip()]
        is_arabic = data.get('isArabic', False)

        try:
            # Generate HTML content without saving
            result = create_result_html(content, query, translated_query, sources, is_arabic)
            return jsonify(result)

        except Exception as html_error:
            logger.error(f"HTML generation failed: {str(html_error)}")
            return jsonify({
                'error': 'Failed to generate HTML',
                'details': str(html_error)
            }), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({
            'error': 'Failed to process request',
            'details': str(e)
        }), 400

if __name__ == '__main__':
    try:
        port = 5000
        logger.info("=" * 60)
        logger.info("HybridRAG Server Starting")
        logger.info("-" * 60)
        logger.info(f"Server URL: http://localhost:{port}")
        logger.info("=" * 60)
        app.run(debug=False, port=port)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise