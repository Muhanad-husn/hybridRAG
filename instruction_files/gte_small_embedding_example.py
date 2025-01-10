# Import necessary libraries from the Hugging Face Transformers and PyTorch
from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model for GTE-Small from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

# Define the input text to be converted into an embedding
text = "Sample text for embedding."

# Tokenize the input text
# - Converts text to token IDs
# - Adds necessary special tokens
# - Pads and truncates the text to fit the model's requirements
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Generate embeddings without computing gradients (inference mode)
with torch.no_grad():
    # Pass tokenized inputs through the model to obtain outputs
    outputs = model(**inputs)
    # Extract the last hidden state tensor (sequence of embeddings for each token)
    token_embeddings = outputs.last_hidden_state

# Compute the mean of the token embeddings to get a single fixed-size vector
# - This represents the entire input text
embedding_vector = token_embeddings.mean(dim=1)

# Print the resulting embedding vector
print(embedding_vector)
