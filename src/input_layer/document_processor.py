import os
import yaml
from typing import List, Dict, Optional, Set
import hashlib
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain.schema import Document
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import pickle
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion and segmentation for multiple file formats with optimized performance."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the document processor with configuration."""
        self.config = self._load_config(config_path)
        self.supported_formats = self.config["document"]["supported_formats"]
        self.llmsherpa_api_url = self.config["llm_sherpa"]["api_url"]
        self.cache_dir = self.config.get("cache", {}).get("directory", "data/cache")
        self.cache_ttl = timedelta(hours=self.config.get("cache", {}).get("ttl_hours", 24))
        self.batch_size = self.config.get("processing", {}).get("batch_size", 1000)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _get_cache_key(self, file_path: str) -> str:
        """Generate a cache key based on file path and modification time."""
        mod_time = os.path.getmtime(file_path)
        return hashlib.md5(f"{file_path}:{mod_time}".encode()).hexdigest()

    def _get_from_cache(self, file_path: str) -> Optional[List[Document]]:
        """Retrieve processed documents from cache if available and not expired."""
        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pickle")
            
            if not os.path.exists(cache_file):
                return None
                
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time > self.cache_ttl:
                os.remove(cache_file)
                return None
                
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {file_path}: {str(e)}")
            return None

    def _save_to_cache(self, file_path: str, docs: List[Document]) -> None:
        """Save processed documents to cache."""
        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pickle")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(docs, f)
        except Exception as e:
            logger.warning(f"Cache save failed for {file_path}: {str(e)}")

    def _is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        file_extension = file_path.split('.')[-1].lower()
        return file_extension in self.supported_formats

    def _process_chunk_batch(self, chunks: List[Document], file_path: str) -> List[Document]:
        """Process a batch of document chunks in parallel."""
        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata['source'] = file_path
            chunk.metadata['file_type'] = file_path.split('.')[-1].lower()
            chunk.metadata['batch_processed'] = datetime.now().isoformat()
        return chunks

    def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file using LLMSherpa with optimized chunk processing."""
        try:
            if not self._is_supported_format(file_path):
                logger.warning(f"Unsupported file format: {file_path}")
                return []

            # Check cache first
            cached_docs = self._get_from_cache(file_path)
            if cached_docs is not None:
                logger.info(f"Retrieved {file_path} from cache")
                return cached_docs

            logger.info(f"Processing file: {file_path}")
            
            # Initialize the LLMSherpa loader with configuration
            loader = LLMSherpaFileLoader(
                file_path=file_path,
                new_indent_parser=True,
                apply_ocr=True,
                strategy="chunks",
                llmsherpa_api_url=self.llmsherpa_api_url
            )

            # Load and process the document
            docs = loader.load()
            
            # Process chunks in batches using parallel processing
            processed_docs = []
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i:i + self.batch_size]
                processed_batch = self._process_chunk_batch(batch, file_path)
                processed_docs.extend(processed_batch)

            logger.info(f"Successfully processed {file_path}: {len(processed_docs)} chunks created")
            
            # Cache the results
            self._save_to_cache(file_path, processed_docs)
            
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def _save_chunk_batch(self, batch: List[Document], output_dir: str, start_idx: int) -> None:
        """Save a batch of chunks to disk."""
        for idx, doc in enumerate(batch, start=start_idx):
            output_path = os.path.join(
                output_dir,
                f"chunk_{idx}_{os.path.basename(doc.metadata['source'])}.txt"
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)

    def process_directory(self, directory_path: str, max_workers: Optional[int] = None) -> List[Document]:
        """
        Process all supported documents in a directory using parallel processing.
        
        Args:
            directory_path: Path to the directory containing documents
            max_workers: Maximum number of parallel workers (defaults to CPU count)
            
        Returns:
            List of processed documents
        """
        try:
            # Get all files in the directory
            files = [
                os.path.join(directory_path, f)
                for f in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, f))
            ]
            
            # Filter for supported files
            supported_files = [f for f in files if self._is_supported_format(f)]
            
            if not supported_files:
                logger.warning(f"No supported files found in {directory_path}")
                return []

            # Process files using both process pool (for CPU-intensive tasks) and thread pool (for I/O)
            processed_docs = []
            with ProcessPoolExecutor(max_workers=max_workers) as process_executor:
                with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
                    # Submit CPU-intensive tasks to process pool
                    future_to_file = {
                        process_executor.submit(self._process_single_file, file_path): file_path
                        for file_path in supported_files
                    }
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            docs = future.result()
                            processed_docs.extend(docs)
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {str(e)}")

            logger.info(f"Completed processing {len(supported_files)} files")
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return []

    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed documents
        """
        return self._process_single_file(file_path)

    def save_processed_chunks(self, documents: List[Document], output_dir: str) -> None:
        """
        Save processed document chunks to the specified directory using parallel processing.
        
        Args:
            documents: List of processed documents
            output_dir: Directory to save the chunks
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save chunks in parallel batches
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    futures.append(
                        executor.submit(self._save_chunk_batch, batch, output_dir, i)
                    )
                
                # Wait for all saves to complete
                for future in as_completed(futures):
                    future.result()  # This will raise any exceptions that occurred
                    
            logger.info(f"Saved {len(documents)} chunks to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to {output_dir}: {str(e)}")
            raise