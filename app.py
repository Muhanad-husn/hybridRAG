from flask import Flask, render_template, request, jsonify
import sys
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator

app = Flask(__name__)
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
            # Translate query to English
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query)
        else:
            result = run_hybrid_search(query)
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)