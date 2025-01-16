from flask import Flask, render_template, request, jsonify, send_file
import sys
import logging
import io
import os
from datetime import datetime
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger

# Configure logging with UTF-8 support
def setup_utf8_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger with UTF-8 support
logger = setup_utf8_logger()

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

# Initialize translator
translator = Translator()

def verify_font():
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', 'fonts', 'NotoNaskhArabic-Regular.ttf')
        logger.info(f"Verifying font at: {font_path}")
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found at {font_path}")
            
        logger.info("Arabic font verified successfully")
    except Exception as e:
        logger.error(f"Font verification failed: {str(e)}")
        raise

# Verify font
verify_font()

def create_result_html(content, query, translated_query, sources, is_arabic=False):
    try:
        # Create timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Render HTML template
        html = render_template(
            'result_template.html',
            content=content,
            query=query,
            translated_query=translated_query,
            sources=sources,
            is_arabic=is_arabic,
            timestamp=timestamp
        )
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"HybridRAG_Result_{'Arabic' if is_arabic else 'English'}_{timestamp_str}.html"
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

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Detect if query is Arabic
        is_arabic = translator.is_arabic(query)
        
        if is_arabic:
            logger.info(f"Processing Arabic query: {query}")
            # Translate query to English for internal processing only
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            logger.info(f"Translated to English for processing: {english_query}")
            # Pass original query without translation
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query)
        else:
            logger.info(f"Processing English query: {query}")
            result = run_hybrid_search(query)
        
        logger.debug("Response data: %s", result)
        logger.debug("Response keys: %s", result.keys())
        logger.debug("Language: %s", result.get('language'))
            
        return jsonify(result)
        
    except Exception as e:
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
        logger.info(f"Number of sources: {len(sources)}")

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