# Import necessary libraries
from sentence_transformers import SentenceTransformer, util
from flashrank import Ranker
import numpy as np

# Sample documents
documents = [
    "Python is a popular programming language.",
    "Data science involves statistics and programming.",
    "Machine learning is a subset of artificial intelligence.",
    "Artificial intelligence is transforming various industries.",
    "Governance and data can solve social problems."
]

# User query
query = "How can data science help in governance?"

# Step 1: Load a pre-trained sentence transformer model
# This model is used to generate embeddings for both documents and the query
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Encode documents and query into embeddings
# Embeddings are vector representations of text for similarity computation
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Step 3: Compute cosine similarities between the query and each document
# Cosine similarity measures the similarity between two vectors
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# Step 4: Combine documents with their similarity scores
# Pair each document with its respective cosine similarity score
doc_scores = list(zip(documents, cosine_scores.cpu().numpy()))

# Step 5: Sort documents based on similarity scores in descending order
# This gives an initial ranking of documents based on their relevance to the query
initial_ranking = sorted(doc_scores, key=lambda x: x[1], reverse=True)

# Step 6: Initialize the Ranker
# The Ranker uses a more sophisticated model to refine the ranking
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Step 7: Prepare documents for reranking
# Convert the initial ranking into the format required by the Ranker
rerank_docs = [{"text": doc} for doc, _ in initial_ranking]

# Step 8: Perform reranking
# The Ranker re-evaluates the relevance of documents and adjusts their order
reranked_results = ranker.rerank(query=query, documents=rerank_docs)

# Step 9: Extract the text content of the reranked documents
# We focus only on the "text" field of the reranked results
reranked_docs = [doc['text'] for doc in reranked_results]

# Step 10: Define a function to generate a response based on the query and context
# This function simulates using the top-ranked documents to generate a response
def generate_response(query, context):
    return f"Response to '{query}' with context: {context}"

# Step 11: Use top N reranked documents as context
# Combine the top N documents to form the context for the response
top_n = 3
context = " ".join(reranked_docs[:top_n])

# Step 12: Generate and display the final response
response = generate_response(query, context)
print(response)
