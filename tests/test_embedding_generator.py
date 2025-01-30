import pytest
from src.processing_layer.embedding_generator import EmbeddingGenerator

@pytest.fixture
def embedding_generator():
    return EmbeddingGenerator()

def test_embedding_generation(embedding_generator):
    test_text = "This is a test sentence"
    embedding = embedding_generator.generate_embedding(test_text)
    assert embedding is not None
    assert len(embedding) == 384  # GTE-small dimension

def test_batch_embeddings(embedding_generator):
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedding_generator.generate_embeddings_batch(texts)
    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)