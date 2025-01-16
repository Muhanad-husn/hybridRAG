from flask import Flask, render_template, request, jsonify, send_file
import sys
import logging
import io
import os
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
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
        self.margin = 20
        self.effective_page_width = 210 - 2 * self.margin  # A4 width = 210mm
        
    def add_arabic_text(self, x, y, txt, align='R'):
        try:
            if not txt:
                return y
            reshaped_text = arabic_reshaper.reshape(txt)
            bidi_text = get_display(reshaped_text)
            
            # Calculate position based on alignment
            if align == 'R':
                x = 210 - self.margin - self.get_string_width(bidi_text)  # A4 width = 210mm
            elif align == 'C':
                x = (210 - self.get_string_width(bidi_text)) / 2
            
            self.text(x, y, bidi_text)
            return y + self.font_size * 1.5
        except Exception as e:
            logger.error(f"Error adding Arabic text: {str(e)}")
            raise

    def add_wrapped_text(self, text, y, is_arabic=False, align='L'):
        try:
            if not text:
                return y
            
            # Split text into words
            words = text.split()
            line = []
            for word in words:
                line.append(word)
                test_line = ' '.join(line)
                
                # For Arabic text, reshape and calculate width
                if is_arabic:
                    test_line = get_display(arabic_reshaper.reshape(test_line))
                
                # Check if line exceeds width
                if self.get_string_width(test_line) > self.effective_page_width:
                    line.pop()  # Remove last word
                    # Print current line
                    current_line = ' '.join(line)
                    if is_arabic:
                        y = self.add_arabic_text(self.margin, y, current_line, align)
                    else:
                        self.text(self.margin, y, current_line)
                        y += self.font_size * 1.5
                    line = [word]  # Start new line with current word
                    
                    # Check if we need a new page
                    if y > 277:  # A4 height = 297mm, leave 20mm margin
                        self.add_page()
                        y = self.margin
            
            # Print remaining words
            if line:
                current_line = ' '.join(line)
                if is_arabic:
                    y = self.add_arabic_text(self.margin, y, current_line, align)
                else:
                    self.text(self.margin, y, current_line)
                    y += self.font_size * 1.5
            
            return y
        except Exception as e:
            logger.error(f"Error in text wrapping: {str(e)}")
            raise

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
            
        # Test font loading with FPDF
        test_pdf = PDF()
        test_pdf.add_page()
        test_pdf.set_font('NotoNaskh', '', 14)
        
        # Test Arabic text rendering
        test_text = "اختبار"
        test_pdf.add_arabic_text(10, 10, test_text)
        
        logger.info("Arabic font verified successfully")
    except Exception as e:
        logger.error(f"Font verification failed: {str(e)}")
        raise

# Verify font
verify_font()

def create_pdf(content, query, translated_query, sources, is_arabic=False):
    try:
        # Initialize PDF
        pdf = PDF()
        pdf.add_page()
        
        # Set initial font and size
        font_size = 12
        pdf.set_font('NotoNaskh' if is_arabic else 'Arial', '', font_size)
        
        # Start position
        y = pdf.margin
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_text = "تم إنشاؤه في: " + timestamp if is_arabic else f"Generated: {timestamp}"
        if is_arabic:
            y = pdf.add_arabic_text(pdf.margin, y, timestamp_text)
        else:
            pdf.text(pdf.margin, y, timestamp_text)
            y += font_size * 1.5
        
        y += font_size  # Add spacing
        
        # Add query section
        if is_arabic:
            # Add Arabic query
            y = pdf.add_arabic_text(pdf.margin, y, "السؤال:")
            y = pdf.add_wrapped_text(query, y, is_arabic=True, align='R')
            
            if translated_query:
                y += font_size
                pdf.set_font('Arial', '', font_size)
                pdf.text(pdf.margin, y, "English Query:")
                y += font_size * 1.5
                y = pdf.add_wrapped_text(translated_query, y)
                pdf.set_font('NotoNaskh', '', font_size)
        else:
            # Add English query
            pdf.text(pdf.margin, y, "Query:")
            y += font_size * 1.5
            y = pdf.add_wrapped_text(query, y)
            
            if translated_query:
                y += font_size
                pdf.set_font('NotoNaskh', '', font_size)
                y = pdf.add_arabic_text(pdf.margin, y, "الترجمة:")
                y = pdf.add_wrapped_text(translated_query, y, is_arabic=True, align='R')
                pdf.set_font('Arial', '', font_size)
        
        y += font_size  # Add spacing
        
        # Add content section
        if is_arabic:
            y = pdf.add_arabic_text(pdf.margin, y, "الإجابة:")
            y = pdf.add_wrapped_text(content, y, is_arabic=True, align='R')
        else:
            pdf.text(pdf.margin, y, "Answer:")
            y += font_size * 1.5
            y = pdf.add_wrapped_text(content, y)
        
        # Add sources section
        if sources:
            y += font_size * 2  # Add extra spacing before sources
            
            if is_arabic:
                y = pdf.add_arabic_text(pdf.margin, y, "المصادر:")
                y += font_size * 1.5
                for i, source in enumerate(sources, 1):
                    bullet_text = f"●  {source}"
                    y = pdf.add_wrapped_text(bullet_text, y, is_arabic=True, align='R')
                    y += font_size * 0.5  # Add small spacing between sources
            else:
                pdf.text(pdf.margin, y, "Sources:")
                y += font_size * 1.5
                for i, source in enumerate(sources, 1):
                    source_text = f"{i}. {source}"
                    y = pdf.add_wrapped_text(source_text, y)
                    y += font_size * 0.5  # Add small spacing between sources
        
        # Save to buffer
        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
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