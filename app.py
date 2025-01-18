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

_font_verified = False
def verify_font():
    global _font_verified
    if _font_verified:
        return
        
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', 'fonts', 'NotoNaskhArabic-Regular.ttf')
        logger.info(f"Verifying font at: {font_path}")
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found at {font_path}")
            
        logger.info("Arabic font verified successfully")
        _font_verified = True
    except Exception as e:
        logger.error(f"Font verification failed: {str(e)}")
        raise

# Verify font
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
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), 'docs')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return filepath
        
    except Exception as e:
        logger.error(f"Error generating HTML: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search-history', methods=['GET'])
def get_search_history():
    return jsonify({'history': search_history})

@app.route('/saved-results', methods=['GET'])
def get_saved_results():
    return jsonify({'results': saved_results})

@app.route('/save-result', methods=['POST'])
def save_result():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
            
        # Add to saved results (maintain uniqueness and limit to 25)
        if filename not in saved_results:
            if len(saved_results) >= 25:
                saved_results.pop()  # Remove oldest entry
            saved_results.insert(0, filename)  # Add new entry at the beginning
            
        return jsonify({'message': 'Result saved successfully', 'saved_results': saved_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
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
                    'Confidence'
                ]):
                    # Extract timestamp and message
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        timestamp = parts[0].strip()
                        message = parts[2].strip()
                        
                        # Create unique key from message content
                        msg_key = message.split(':', 1)[0] if ':' in message else message
                        
                        # Add to deque with timestamp
                        lines.append(f"[{timestamp}] {message}")
            
            # Convert deque to list and remove duplicates while preserving order
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
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Update search history (maintain uniqueness and limit to 25)
        if query not in search_history:
            if len(search_history) >= 25:
                search_history.pop()  # Remove oldest entry
            search_history.insert(0, query)  # Add new query at the beginning

        # Detect if query is Arabic
        is_arabic = translator.is_arabic(query)
        
        if is_arabic:
            logger.info(f"Processing Arabic query: {query}")
            # Translate query to English for internal processing only
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            # Pass original query without translation
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query)
        else:
            logger.info(f"Processing English query: {query}")
            result = run_hybrid_search(query)

        # Get the most recent HTML files from docs directory
        docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
        html_files = [f for f in os.listdir(docs_dir) if f.endswith('.html')]
        html_files.sort(key=lambda x: os.path.getmtime(os.path.join(docs_dir, x)), reverse=True)

        # Get the latest file
        if html_files:
            result['result_file'] = html_files[0]
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
def serve_result(filename):
    try:
        docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
        filepath = os.path.join(docs_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({'error': 'File not found'}), 404
            
        logger.info(f"Serving file: {filepath}")
        
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
def generate_result():
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract and validate content
        content = data.get('content', '').strip()
        if not content:
            return jsonify({'error': 'Content cannot be empty'}), 400

        # Extract other fields with defaults
        query = data.get('query', '').strip()
        translated_query = data.get('translatedQuery', '').strip()
        sources = [s for s in data.get('sources', []) if s.strip()]
        is_arabic = data.get('isArabic', False)

        # Log basic info without Arabic text to avoid encoding issues
        logger.info(f"Starting HTML generation - Language: {'Arabic' if is_arabic else 'English'}")
        logger.info(f"Content length: {len(content)}")

        try:
            # Generate HTML file
            filepath = create_result_html(content, query, translated_query, sources, is_arabic)

            # Return HTML file
            return send_file(
                filepath,
                mimetype='text/html',
                as_attachment=True,
                download_name=os.path.basename(filepath)
            )

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
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise