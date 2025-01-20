import os
import yaml
import torch
import numpy as np
import faiss
import shutil
import traceback
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddings(Embeddings):
    """Custom embeddings class for use with LangChain."""
    
    def __init__(self, embedding_generator):
        """Initialize with reference to EmbeddingGenerator."""
        self.generator = embedding_generator
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        return [self.generator.generate_embedding(text).tolist() for text in texts]
        
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query text."""
        return self.generator.generate_embedding(text).tolist()

class EmbeddingGenerator:
    """Generates embeddings for documents using the GTE-Small model."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the embedding generator with configuration."""
        self.config = self._load_config(config_path)
        self.model_name = self.config["embedding"]["model_name"]
        self.max_length = self.config["embedding"]["max_length"]
        self.cache_dir = self.config["embedding"]["cache_dir"]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create embeddings directory if it doesn't exist
        self.embeddings_dir = self.config["paths"]["embeddings"]
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize tokenizer and model first
        self._initialize_model()
        
        # Initialize custom embeddings
        self.embedding_function = CustomEmbeddings(self)
        
        # Initialize vector store
        self._initialize_vector_store()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _initialize_model(self) -> None:
        """Initialize the tokenizer and model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize or load the vector store."""
        try:
            # Set up storage directory
            os.makedirs(self.embeddings_dir, exist_ok=True)
            index_path = os.path.join(self.embeddings_dir, 'index.faiss')

            # Load existing index if it exists
            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(
                    self.embeddings_dir,
                    self.embedding_function,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing vector store")
            else:
                # Create new empty vector store
                self._create_new_vector_store()
                logger.info("Created new vector store")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
            
    def _create_new_vector_store(self) -> None:
        """Create a new empty vector store."""
        try:
            logger.info("Creating new FAISS index")
            dimension = 384  # GTE-small embedding dimension
            
            # Create and configure FAISS index
            index = faiss.IndexFlatIP(dimension)
            if faiss.get_num_gpus() > 0:
                logger.info("Moving index to GPU")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # Create vector store with the index
            self.vector_store = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            
            # Test the empty index
            test_query = "test query"
            test_embedding = self.generate_embedding(test_query)
            if test_embedding is None or len(test_embedding) != dimension:
                raise ValueError(f"Invalid test embedding dimension: {len(test_embedding) if test_embedding is not None else 'None'}")
            
            logger.info("Vector store initialized and tested successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def reset_vector_store(self) -> None:
        """Reset the vector store by creating a new empty FAISS index."""
        try:
            # Set up storage directory
            os.makedirs(self.embeddings_dir, exist_ok=True)
            
            # Clean up old files first
            for item in os.listdir(self.embeddings_dir):
                item_path = os.path.join(self.embeddings_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            # Create new empty FAISS index
            dimension = 384  # GTE-small embedding dimension
            index = faiss.IndexFlatIP(dimension)
            
            # Create new FAISS vector store
            self.vector_store = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            
            # Save the empty index
            self.vector_store.save_local(self.embeddings_dir)
            
            # Verify the index was saved
            if not os.path.exists(os.path.join(self.embeddings_dir, 'index.faiss')):
                raise ValueError("Failed to save FAISS index")
            
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            raise
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text input.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array containing the embedding
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert to numpy and normalize
            embedding = embeddings[0].cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
            

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_workers: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts using parallel processing.
        
        Args:
            texts: List of input texts
            batch_size: Size of batches for processing
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of numpy arrays containing the embeddings
        """
        try:
            embeddings = []
            
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_text = {
                        executor.submit(self.generate_embedding, text): text
                        for text in batch_texts
                    }
                    
                    for future in as_completed(future_to_text):
                        text = future_to_text[future]
                        try:
                            embedding = future.result()
                            embeddings.append(embedding)
                        except Exception as e:
                            logger.error(f"Error processing text: {str(e)}")
                            # Append zero vector as fallback
                            embeddings.append(np.zeros(self.model.config.hidden_size))
                
                logger.info(f"Processed batch {i//batch_size + 1}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise

    def process_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        max_workers: Optional[int] = None
    ) -> List[Document]:
        """
        Generate embeddings for a list of documents and add them to metadata.
        
        Args:
            documents: List of input documents
            batch_size: Size of batches for processing
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of documents with embeddings added to metadata
        """
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(
                texts,
                batch_size=batch_size,
                max_workers=max_workers
            )
            
            # Add embeddings to document metadata
            for doc, embedding in zip(documents, embeddings):
                doc.metadata['embedding'] = embedding
            
            logger.info(f"Generated embeddings for {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def save_embeddings(
        self,
        documents: List[Document]
    ) -> None:
        """
        Save document embeddings to the vector store.
        
        Args:
            documents: List of documents with embeddings
        """
        try:
            # Extract texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for idx, doc in enumerate(documents):
                texts.append(doc.page_content)
                
                # Create unique document ID using source and chunk index
                doc_id = f"doc_{idx}"
                if 'source' in doc.metadata:
                    base_name = os.path.basename(doc.metadata['source'])
                    doc_id = f"{os.path.splitext(base_name)[0]}_chunk_{idx}"
                ids.append(doc_id)
                
                # Store all metadata except the embedding itself
                meta = {k: v for k, v in doc.metadata.items() if k != 'embedding'}
                metadatas.append(meta)
            
            # Add documents to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Saved {len(documents)} embeddings to vector store")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
            
    def load_embeddings(self, query: str = "", k: int = 100) -> List[Document]:
        """
        Load embeddings from the vector store.
        
        Args:
            query: Query text to find similar documents. If empty, returns random documents.
            k: Number of documents to return
            
        Returns:
            List[Document]: List of documents with embeddings
        """
        try:
            # Use similarity search to retrieve documents
            results = self.vector_store.similarity_search_with_score(
                query if query else ".",  # Use "." as default query to get random docs
                k=k
            )
            
            # Extract just the documents from (doc, score) tuples
            documents = [doc for doc, _ in results]
            
            logger.info(f"Loaded {len(documents)} embeddings from vector store")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise