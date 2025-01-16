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
from src.utils.logger import setup_logger

import sys
import logging
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator
from src.utils.logger import setup_logger, get_logger

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses

# Initialize logger
setup_logger()  # This will set up the root logger with config
logger = get_logger(__name__)  # Get a logger for this module
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 encoding for JSON responses
translator = Translator()

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

# Register Arabic font
font_path = os.path.join(app.static_folder, 'assets', 'fonts', 'NotoNaskhArabic-Regular.ttf')
pdfmetrics.registerFont(TTFont('NotoNaskh', font_path))

def create_pdf(content, query, sources, is_arabic=False):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Set font and size
    p.setFont('NotoNaskh' if is_arabic else 'Helvetica', 12)
    
    # Initialize positions
    margin = 50
    y = height - margin
    line_height = 20
    max_width = width - 2 * margin
    
    def add_text_line(text, y_pos):
        if is_arabic:
            # Reshape and reorder Arabic text
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            p.drawRightString(width - margin, y_pos, bidi_text)
        else:
            p.drawString(margin, y_pos, text)
        return y_pos - line_height

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y = add_text_line(f"{'تم إنشاؤه في' if is_arabic else 'Generated at'}: {timestamp}", y)
    y -= line_height

    # Add query
    y = add_text_line(f"{'السؤال' if is_arabic else 'Query'}:", y)
    for line in query.split('\n'):
        y = add_text_line(line, y)
    y -= line_height

    # Add content
    y = add_text_line(f"{'الإجابة' if is_arabic else 'Answer'}:", y)
    # Split content into lines that fit the page width
    words = content.split()
    current_line = []
    for word in words:
        current_line.append(word)
        line_text = ' '.join(current_line)
        if p.stringWidth(line_text, p._fontname, p._fontsize) > max_width:
            current_line.pop()
            y = add_text_line(' '.join(current_line), y)
            current_line = [word]
            
            # Check if we need a new page
            if y < margin:
                p.showPage()
                p.setFont('NotoNaskh' if is_arabic else 'Helvetica', 12)
                y = height - margin
    
    if current_line:
        y = add_text_line(' '.join(current_line), y)
    y -= line_height

    # Add sources
    if sources:
        y = add_text_line(f"{'المصادر' if is_arabic else 'Sources'}:", y)
        for i, source in enumerate(sources, 1):
            source_text = f"{source} .{i}" if is_arabic else f"{i}. {source}"
            y = add_text_line(source_text, y)
            if y < margin:
                p.showPage()
                p.setFont('NotoNaskh' if is_arabic else 'Helvetica', 12)
                y = height - margin

    p.save()
    buffer.seek(0)
    return buffer

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        query = data.get('query', '').strip()
        sources = data.get('sources', [])
        is_arabic = data.get('isArabic', False)
        
        if not content:
            return jsonify({'error': 'Content cannot be empty'}), 400

        logger.info(f"Generating PDF for {'Arabic' if is_arabic else 'English'} content")
        pdf_buffer = create_pdf(content, query, sources, is_arabic)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"HybridRAG_Result_{'Arabic' if is_arabic else 'English'}_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        translator = Translator()
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Failed to initialize translator: {str(e)}")
        raise