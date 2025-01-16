from flask import Flask, render_template, request, jsonify
import sys
from retrieve_syria import run_hybrid_search
from src.input_layer.translator import Translator

app = Flask(__name__)
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
            print(f"Processing Arabic query: {query}")
            # Translate query to English
            english_query = translator.translate(query, source_lang='ar', target_lang='en')
            print(f"Translated to English: {english_query}")
            result = run_hybrid_search(english_query, original_lang='ar', original_query=query)
        else:
            print(f"Processing English query: {query}")
            result = run_hybrid_search(query)
        
        print("Response data:", result)
        print("Response keys:", result.keys())
        print("Language:", result.get('language'))
        print("Has answer:", 'answer' in result)
        print("Has english_answer:", 'english_answer' in result)
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        translator = Translator()
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to initialize translator: {str(e)}")
        raise