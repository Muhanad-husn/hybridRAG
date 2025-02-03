import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from langchain.schema import Document
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.utils.config_handler import config
from src.utils.cache_handler import DocumentCache
from src.utils.error_handler import (
    log_errors,
    DocumentProcessingError,
    InvalidFormatError,
    ProcessingError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MetadataBuilder:
    @staticmethod
    def from_file(file_path: str) -> Dict:
        return {
            'source': file_path,
            'file_type': Path(file_path).suffix[1:].lower(),
            'processed_at': datetime.now().isoformat()
        }

def parallel_batch_process(batch_size: int = 1000):
    """Decorator for parallel batch processing."""
    def decorator(func):
        def wrapper(self, chunks: List[Document], *args, **kwargs):
            processed_chunks = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    futures.append(
                        executor.submit(func, self, batch, *args, **kwargs)
                    )
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_chunks.extend(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {str(e)}")
                        raise ProcessingError(f"Batch processing failed: {str(e)}")
            return processed_chunks
        return wrapper
    return decorator

class DocumentProcessor:
    """Handles document ingestion and segmentation for multiple file formats with optimized performance."""
    
    def __init__(self):
        """Initialize the document processor with configuration."""
        self.supported_formats = config.get("document.supported_formats", [])
        self.batch_size = config.get("processing.batch_size", 1000)
        self.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-small")
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=75, 
            embed_model=self.embed_model
        )
        
        # Initialize cache with config
        self.cache = DocumentCache[List[Document]](
            cache_dir=config.get("cache.directory", "data/cache"),
            ttl_hours=config.get("cache.ttl_hours", 24)
        )

    def _is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        file_extension = Path(file_path).suffix[1:].lower()
        if file_extension not in self.supported_formats:
            raise InvalidFormatError(
                f"Unsupported file format: {file_extension}",
                {"supported_formats": self.supported_formats}
            )
        return True

    @parallel_batch_process(batch_size=1000)
    def _process_chunk_batch(self, chunks: List[Document], file_path: str) -> List[Document]:
        """Process a batch of document chunks in parallel."""
        try:
            for chunk in chunks:
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                chunk.metadata.update(MetadataBuilder.from_file(file_path))
            return chunks
        except Exception as e:
            raise ProcessingError(f"Chunk processing failed: {str(e)}")

    @log_errors(logger)
    def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file using llama_index with optimized chunk processing."""
        try:
            self._is_supported_format(file_path)

            # Check cache first
            cached_docs = self.cache.get(file_path)
            if cached_docs is not None:
                logger.info(f"Retrieved {file_path} from cache")
                return cached_docs

            logger.info(f"Processing file: {file_path}")
            
            # New document loading and processing
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            nodes = self.splitter.get_nodes_from_documents(documents)
            
            # Convert nodes to Documents and add metadata
            processed_docs = []
            for node in nodes:
                doc = Document(page_content=node.text, metadata=node.metadata)
                doc.metadata.update(MetadataBuilder.from_file(file_path))
                processed_docs.append(doc)
            
            # Process chunks in batches using parallel processing
            processed_docs = self._process_chunk_batch(processed_docs, file_path)
            
            logger.info(f"Successfully processed {file_path}: {len(processed_docs)} chunks created")
            
            # Cache the results with metadata
            self.cache.set(
                file_path, 
                processed_docs,
                extra_metadata=MetadataBuilder.from_file(file_path)
            )
            
            return processed_docs

        except InvalidFormatError:
            logger.warning(f"Skipping unsupported file: {file_path}")
            return []
        except Exception as e:
            raise ProcessingError(f"File processing failed: {str(e)}")

    @log_errors(logger)
    def process_directory(self, directory_path: str, max_workers: Optional[int] = None) -> List[Document]:
        """Process all supported documents in a directory using parallel processing."""
        try:
            # Get all files in the directory
            files = [
                os.path.join(directory_path, f)
                for f in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, f))
            ]
            
            # Filter for supported files
            supported_files = []
            for file_path in files:
                try:
                    if self._is_supported_format(file_path):
                        supported_files.append(file_path)
                except InvalidFormatError:
                    continue
            
            if not supported_files:
                logger.warning(f"No supported files found in {directory_path}")
                return []

            # Process files using process pool for CPU-intensive tasks
            processed_docs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
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
            raise ProcessingError(f"Directory processing failed: {str(e)}")

    @log_errors(logger)
    def process_file(self, file_path: str) -> List[Document]:
        """Process a single document file."""
        return self._process_single_file(file_path)

    @log_errors(logger)
    def save_processed_chunks(self, documents: List[Document], output_dir: str) -> None:
        """Save processed document chunks to the specified directory using parallel processing."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            def save_chunk_batch(batch: List[Document], start_idx: int) -> None:
                for idx, doc in enumerate(batch, start=start_idx):
                    output_path = os.path.join(
                        output_dir,
                        f"chunk_{idx}_{Path(doc.metadata['source']).name}.txt"
                    )
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(doc.page_content)
            
            # Save chunks in parallel batches
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    futures.append(
                        executor.submit(save_chunk_batch, batch, i)
                    )
                
                # Wait for all saves to complete
                for future in as_completed(futures):
                    future.result()  # This will raise any exceptions that occurred
                    
            logger.info(f"Saved {len(documents)} chunks to {output_dir}")
            
        except Exception as e:
            raise ProcessingError(f"Error saving chunks: {str(e)}")