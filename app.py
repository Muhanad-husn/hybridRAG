from flask import Flask, render_template, request, jsonify, send_file
import sys
import logging
import io
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

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

# Initialize translator
translator = Translator()

# Configure Flask app
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

def register_font():
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', 'fonts', 'NotoNaskhArabic-Regular.ttf')
        logger.info(f"Attempting to load font from: {font_path}")
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found at {font_path}")
            
        pdfmetrics.registerFont(TTFont('NotoNaskh', font_path))
        logger.info("Arabic font registered successfully")
    except Exception as e:
        logger.error(f"Failed to register Arabic font: {str(e)}")
        raise

# Register font
register_font()

def create_pdf(content, query, translated_query, sources, is_arabic=False):
    buffer = io.BytesIO()
    doc = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    # Basic PDF generation with minimal formatting
    try:
        # Set basic font
        doc.setFont('Helvetica', 12)
        
        # Add timestamp
        doc.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 20
        
        # Add query
        doc.drawString(margin, y, "Query:")
        y -= 20
        doc.drawString(margin, y, query)
        y -= 20
        
        if translated_query:
            doc.drawString(margin, y, "Translated Query:")
            y -= 20
            doc.drawString(margin, y, translated_query)
            y -= 20
        
        # Add content
        doc.drawString(margin, y, "Answer:")
        y -= 20
        
        # Simple text wrapping
        words = content.split()
        line = []
        for word in words:
            line.append(word)
            if doc.stringWidth(' '.join(line)) > width - 2 * margin:
                line.pop()
                doc.drawString(margin, y, ' '.join(line))
                y -= 15
                line = [word]
                
                if y < margin:
                    doc.showPage()
                    doc.setFont('Helvetica', 12)
                    y = height - margin
        
        if line:
            doc.drawString(margin, y, ' '.join(line))
            y -= 20
        
        # Add sources
        if sources:
            doc.drawString(margin, y, "Sources:")
            y -= 20
            for i, source in enumerate(sources, 1):
                if source.strip():
                    doc.drawString(margin, y, f"{i}. {source}")
                    y -= 15
                    if y < margin:
                        doc.showPage()
                        doc.setFont('Helvetica', 12)
                        y = height - margin
        
        doc.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise

    p.save()
    buffer.seek(0)
    return buffer

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
            # Translate query to English
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            logger.info(f"Translated to English: {english_query}")
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

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
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
        logger.info(f"Starting PDF generation - Language: {'Arabic' if is_arabic else 'English'}")
        logger.info(f"Content length: {len(content)}")
        logger.info(f"Number of sources: {len(sources)}")

        try:
            # Generate PDF
            pdf_buffer = create_pdf(content, query, translated_query, sources, is_arabic)

            # Create filename with unique timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"HybridRAG_Result_{'Arabic' if is_arabic else 'English'}_{timestamp}.pdf"

            # Return PDF file with cache prevention headers
            response = send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            logger.info("PDF generated successfully")
            return response

        except Exception as pdf_error:
            logger.error(f"PDF generation failed: {str(pdf_error)}")
            return jsonify({
                'error': 'Failed to generate PDF',
                'details': str(pdf_error)
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