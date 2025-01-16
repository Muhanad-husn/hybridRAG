from flask import Flask, render_template, request, jsonify, send_file
import sys
import logging
import io
import os
from datetime import datetime
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger

# Custom PDF class with Arabic support
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('NotoNaskh', '', 'static/assets/fonts/NotoNaskhArabic-Regular.ttf', uni=True)
        
    def add_arabic_text(self, x, y, txt):
        reshaped_text = arabic_reshaper.reshape(txt)
        bidi_text = get_display(reshaped_text)
        self.text(x, y, bidi_text)

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
    try:
        # Initialize PDF
        pdf = PDF()
        pdf.add_page()
        margin = 20
        
        # Set initial font
        if is_arabic:
            pdf.set_font('NotoNaskh', '', 14)
            pdf.set_right_margin(margin)
            pdf.set_left_margin(margin)
        else:
            pdf.set_font('Arial', '', 12)
            pdf.set_left_margin(margin)
            pdf.set_right_margin(margin)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if is_arabic:
            pdf.add_arabic_text(190 - pdf.get_string_width("تم إنشاؤه في: " + timestamp), 10, "تم إنشاؤه في: " + timestamp)
        else:
            pdf.text(margin, 10, f"Generated: {timestamp}")
        
        # Add query section
        y = 30
        if is_arabic:
            # Add Arabic query
            pdf.add_arabic_text(190 - pdf.get_string_width("السؤال:"), y, "السؤال:")
            y += 10
            # Word wrap for Arabic query
            lines = pdf.multi_cell(0, 10, query, split_only=True)
            for line in lines:
                pdf.add_arabic_text(190 - pdf.get_string_width(line), y, line)
                y += 10
            
            if translated_query:
                y += 5
                pdf.set_font('Arial', '', 12)
                pdf.text(margin, y, "English Query:")
                y += 10
                pdf.multi_cell(0, 10, translated_query)
                y += 5
                pdf.set_font('NotoNaskh', '', 14)
        else:
            # Add English query
            pdf.text(margin, y, "Query:")
            y += 10
            pdf.multi_cell(0, 10, query)
            y = pdf.get_y() + 5
            
            if translated_query:
                pdf.set_font('NotoNaskh', '', 14)
                pdf.add_arabic_text(190 - pdf.get_string_width("الترجمة:"), y, "الترجمة:")
                y += 10
                lines = pdf.multi_cell(0, 10, translated_query, split_only=True)
                for line in lines:
                    pdf.add_arabic_text(190 - pdf.get_string_width(line), y, line)
                    y += 10
                pdf.set_font('Arial', '', 12)
        
        # Add content section
        y = pdf.get_y() + 10
        if is_arabic:
            pdf.add_arabic_text(190 - pdf.get_string_width("الإجابة:"), y, "الإجابة:")
            y += 10
            lines = pdf.multi_cell(0, 10, content, split_only=True)
            for line in lines:
                pdf.add_arabic_text(190 - pdf.get_string_width(line), y, line)
                y += 10
        else:
            pdf.text(margin, y, "Answer:")
            y += 10
            pdf.multi_cell(0, 10, content)
        
        # Add sources section
        if sources:
            y = pdf.get_y() + 10
            if is_arabic:
                pdf.add_arabic_text(190 - pdf.get_string_width("المصادر:"), y, "المصادر:")
                y += 10
                for i, source in enumerate(sources, 1):
                    bullet = f"●  {source}"
                    lines = pdf.multi_cell(0, 10, bullet, split_only=True)
                    for line in lines:
                        pdf.add_arabic_text(190 - pdf.get_string_width(line), y, line)
                        y += 10
            else:
                pdf.text(margin, y, "Sources:")
                y += 10
                for i, source in enumerate(sources, 1):
                    pdf.multi_cell(0, 10, f"{i}. {source}")
        
        # Save to buffer
        buffer = io.BytesIO()
        pdf.output(buffer)
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