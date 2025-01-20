from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_model():
    # Set model name
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Set target directory
    target_dir = os.path.join("models", "ranking_model")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading model {model_name} to {target_dir}")
    
    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save model and tokenizer
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    
    print("Model downloaded successfully")

if __name__ == "__main__":
    download_model()