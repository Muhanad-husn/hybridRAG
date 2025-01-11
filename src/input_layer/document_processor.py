import os
import yaml
from typing import List, Dict, Optional
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain.schema import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion and segmentation for multiple file formats."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the document processor with configuration."""
        self.config = self._load_config(config_path)
        self.supported_formats = self.config["document"]["supported_formats"]
        self.llmsherpa_api_url = self.config["llm_sherpa"]["api_url"]
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        file_extension = file_path.split('.')[-1].lower()
        return file_extension in self.supported_formats

    def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file using LLMSherpa."""
        try:
            if not self._is_supported_format(file_path):
                logger.warning(f"Unsupported file format: {file_path}")
                return []

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
            
            # Add source metadata to each document
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source'] = file_path
                doc.metadata['file_type'] = file_path.split('.')[-1].lower()

            logger.info(f"Successfully processed {file_path}: {len(docs)} chunks created")
            return docs

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

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

            # Process files in parallel
            processed_docs = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        Save processed document chunks to the specified directory.
        
        Args:
            documents: List of processed documents
            output_dir: Directory to save the chunks
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, doc in enumerate(documents):
                output_path = os.path.join(
                    output_dir,
                    f"chunk_{idx}_{os.path.basename(doc.metadata['source'])}.txt"
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(doc.page_content)
                    
            logger.info(f"Saved {len(documents)} chunks to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to {output_dir}: {str(e)}")
            raise