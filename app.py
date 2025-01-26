from flask import Flask, render_template, request, jsonify, send_file
import io
import os
import json
import yaml
from datetime import datetime
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger
from functools import wraps
from Process_files import HyperRAG
import asyncio

def manage_results_directory():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.html')]
    files.sort(key=os.path.getmtime, reverse=True)
    
    while len(files) > 15:
        oldest_file = files.pop()
        os.remove(oldest_file)
        logger.info(f"Removed old result file: {oldest_file}")

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

# Initialize logger using the unified logging setup
setup_logger()
logger = get_logger(__name__)

# Initialize translator and search history
translator = Translator()

def load_search_history():
    try:
        history_file = os.path.join(os.path.dirname(__file__), 'data', 'history', 'search_history.json')
        
        # Create history directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'history'), exist_ok=True)
        
        # Load search history
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                search_history = json.load(f)
        else:
            search_history = []
            
        return search_history
    except Exception as e:
        logger.error(f"Error loading search history: {str(e)}")
        return []

def save_search_history(search_history):
    try:
        history_file = os.path.join(os.path.dirname(__file__), 'data', 'history', 'search_history.json')
        
        # Save search history
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(search_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving search history: {str(e)}")

# Initialize search history from file
search_history = load_search_history()

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
            
        # Ensure results directory exists
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save file to results directory
        file_path = os.path.join(results_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Manage results directory after saving
        manage_results_directory()
            
        # Create HTML response for browser download
        logger.info(f"Saving and sending result for download: {filename}")
        return send_file(
            io.BytesIO(html.encode('utf-8')),
            mimetype='text/html',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error saving result: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-saved-results', methods=['GET'])
@log_request
def get_saved_results():
    try:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
            
        results = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.html'):
                file_path = os.path.join(results_dir, filename)
                results.append({
                    'filename': filename,
                    'timestamp': os.path.getmtime(file_path)
                })
        
        # Sort by timestamp, newest first, and limit to 15
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        results = results[:15]
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error getting saved results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
@log_request
def get_logs():
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            # Get all lines from the log file
            lines = []
            for line in f:
                # Extract timestamp and message
                parts = line.split(' - ')
                if len(parts) >= 3:
                    timestamp = parts[0].strip()
                    message = parts[2].strip()
                    lines.append(f"[{timestamp}] {message}")
            
            # Return the last 50 lines
            unique_logs = lines[-50:]
            
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
            save_search_history(search_history)  # Persist changes

        # Detect if query is Arabic
        is_arabic = translator.is_arabic(query)
        
        # Get rerank count from request, default to 15 if not provided or invalid
        rerank_count = max(min(int(data.get('rerank_count', 15)), 80), 5)
        logger.info(f"Using rerank count: {rerank_count}")

        # Get max_tokens and temperature from request, use default values if not provided
        max_tokens = int(data.get('max_tokens', 3000))
        temperature = float(data.get('temperature', 0.0))
        logger.info(f"Using max_tokens: {max_tokens}, temperature: {temperature}")

        if is_arabic:
            logger.info(f"Processing Arabic query: {query}")
            # Translate query to English for internal processing only
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            # Pass original query without translation and respect translation preference
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query,
                                     translate=translate_enabled, rerank_count=rerank_count,
                                     max_tokens=max_tokens, temperature=temperature)
        else:
            logger.info(f"Processing English query: {query}")
            result = run_hybrid_search(query, translate=translate_enabled, rerank_count=rerank_count,
                                       max_tokens=max_tokens, temperature=temperature)

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

# Initialize HyperRAG with app logger
rag_system = HyperRAG(logger=logger)

@app.route('/upload-files', methods=['POST'])
@log_request
def upload_files():
    try:
        # Get the raw_documents directory path
        raw_docs_dir = os.path.join("data", "raw_documents")
        
        # Get uploaded files
        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return jsonify({'error': 'No files uploaded'}), 400

        # Store files in memory first
        files_to_save = []
        for file in uploaded_files:
            if file.filename:
                # Store file content and name
                content = file.read()
                files_to_save.append((file.filename, content))
                
        if not files_to_save:
            return jsonify({'error': 'No valid files to upload'}), 400
            
        # Now that we have all files, clear the target directory
        for filename in os.listdir(raw_docs_dir):
            file_path = os.path.join(raw_docs_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        # Save all files at once
        for filename, content in files_to_save:
            with open(os.path.join(raw_docs_dir, filename), 'wb') as f:
                f.write(content)
                
        return jsonify({
            'message': 'Files uploaded successfully',
            'count': len(files_to_save)
        })
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process-documents', methods=['POST'])
@log_request
def process_documents():
    try:
        data = request.get_json()
        input_dir = data.get('input_dir', os.path.join("data", "raw_documents"))
        save_chunks = data.get('save_chunks', True)
        save_embeddings = data.get('save_embeddings', True)

        if not os.path.exists(input_dir):
            return jsonify({'error': f'Directory not found: {input_dir}'}), 404

        if not os.listdir(input_dir):
            return jsonify({'error': f'No documents found in {input_dir}'}), 400

        # Reset storage first
        rag_system.reset_storage()
        logger.info("Storage reset completed")

        # Then process documents
        asyncio.run(rag_system.aprocess_documents(
            input_dir=input_dir,
            save_chunks=save_chunks,
            save_embeddings=save_embeddings
        ))

        # Get vector store stats
        vector_count = len(os.listdir(os.path.join("data", "embeddings"))) - 1  # Subtract 1 for index.faiss

        # Get graph stats
        nodes_file = os.path.join("data", "graphs", "nodes.csv")
        edges_file = os.path.join("data", "graphs", "edges.csv")
        
        node_count = 0
        edge_count = 0
        
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r') as f:
                node_count = sum(1 for line in f) - 1  # Subtract 1 for header
                
        if os.path.exists(edges_file):
            with open(edges_file, 'r') as f:
                edge_count = sum(1 for line in f) - 1  # Subtract 1 for header

        return jsonify({
            'message': 'Documents processed successfully',
            'input_dir': input_dir,
            'vector_count': vector_count,
            'node_count': node_count,
            'edge_count': edge_count
        })

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update-api-key', methods=['POST'])
@log_request
def update_api_key():
    try:
        data = request.get_json()
        new_api_key = data.get('api_key', '').strip()
        
        if not new_api_key:
            return jsonify({'error': 'API key cannot be empty'}), 400
            
        if not new_api_key.startswith('sk-or-v1-'):
            return jsonify({'error': 'Invalid OpenRouter API key format'}), 400
            
        # Read current .env content
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        try:
            with open(env_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
            
        # Update or add API key
        api_key_updated = False
        for i, line in enumerate(lines):
            if line.startswith('OPENROUTER_API_KEY='):
                lines[i] = f'OPENROUTER_API_KEY={new_api_key}\n'
                api_key_updated = True
                break
                
        if not api_key_updated:
            lines.append(f'OPENROUTER_API_KEY={new_api_key}\n')
            
        # Write back to .env
        with open(env_path, 'w') as f:
            f.writelines(lines)
            
        logger.info("OpenRouter API key updated successfully")
        return jsonify({'message': 'API key updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update-model-settings', methods=['POST'])
@log_request
def update_model_settings():
    try:
        data = request.get_json()
        extraction_model = data.get('extraction_model', '').strip()
        answer_model = data.get('answer_model', '').strip()
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        
        if not extraction_model or not answer_model or max_tokens is None or temperature is None:
            return jsonify({'error': 'All fields (extraction_model, answer_model, max_tokens, and temperature) must be provided'}), 400
            
        # Read current config
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update model settings
        config['llm']['extraction_model'] = extraction_model
        config['llm']['answer_model'] = answer_model
        config['llm']['max_tokens'] = max_tokens
        config['llm']['temperature'] = temperature
        
        # Write back to config file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Model settings updated - Extraction: {extraction_model}, Answer: {answer_model}, Max Tokens: {max_tokens}, Temperature: {temperature}")
        return jsonify({
            'message': 'Model settings updated successfully',
            'extraction_model': extraction_model,
            'answer_model': answer_model,
            'max_tokens': max_tokens,
            'temperature': temperature
        })
        
    except Exception as e:
        logger.error(f"Error updating model settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        manage_results_directory()  # Manage results directory at startup
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