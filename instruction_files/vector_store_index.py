
# Import necessary modules from LlamaIndex
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import SimpleVectorStore
from llama_index.schema import TextNode

# Define your precomputed embeddings and corresponding text chunks
# Replace 'your_precomputed_embeddings' with your actual embeddings
# and 'your_text_chunks' with the corresponding text data
precomputed_embeddings = [
    # Example: [0.1, 0.2, 0.3, ...]  # Embedding vector for the first text chunk
    # Add your actual embedding vectors here
]

text_chunks = [
    "First text chunk corresponding to the first embedding.",
    "Second text chunk corresponding to the second embedding.",
    # Add your actual text chunks here
]

# Ensure the number of embeddings matches the number of text chunks
assert len(precomputed_embeddings) == len(text_chunks), "Embeddings and text chunks count mismatch."

# Create TextNode objects with embeddings
nodes = [
    TextNode(text=chunk, embedding=embedding)
    for chunk, embedding in zip(text_chunks, precomputed_embeddings)
]

# Initialize an in-memory vector store
vector_store = SimpleVectorStore()

# Insert nodes with precomputed embeddings into the vector store
vector_store.add(nodes)

# Create a storage context with the vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize the VectorStoreIndex with the storage context
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Perform a query using the query engine
# Replace 'Your query here' with the actual question or search term
response = query_engine.query('Your query here')

# Print the response from the query
print(response)
