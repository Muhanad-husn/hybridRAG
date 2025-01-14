import os
import yaml
import logging
import hashlib
from typing import Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from cachetools import TTLCache

# Set seed for consistent language detection
DetectorFactory.seed = 0

class Translator:
    """Handles translation between Arabic and English."""
    
    def __init__(self, config_path: str = "config/translation_config.yaml"):
        """Initialize the translator with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize translation models and tokenizers
        self.ar_to_en_model, self.ar_to_en_tokenizer = self._load_model("ar_to_en")
        self.en_to_ar_model, self.en_to_ar_tokenizer = self._load_model("en_to_ar")
        
        # Initialize cache if enabled
        if self.config["cache"]["enabled"]:
            self.cache = TTLCache(
                maxsize=self.config["cache"]["max_size"],
                ttl=self.config["cache"]["ttl"]
            )
        else:
            self.cache = None
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load translation configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading translation config: {str(e)}")
            raise
            
    def _load_model(self, direction: str) -> tuple[MarianMTModel, MarianTokenizer]:
        """Load translation model and tokenizer for specified direction."""
        try:
            model_config = self.config["models"][direction]
            # Map direction to correct model name
            model_mapping = {
                "ar_to_en": "Helsinki-NLP/opus-mt-ar-en",
                "en_to_ar": "Helsinki-NLP/opus-mt-tc-big-en-ar"  # Using the bigger model for better quality
            }
            
            model_name = model_mapping[direction]
            self.logger.info(f"Loading translation model: {model_name}")
            
            # Load tokenizer and model with error details
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
            
            # Set device
            device = model_config["device"]
            model = model.to(device)
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading {direction} model: {str(e)}")
            raise
            
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation."""
        key_string = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(key_string.encode()).hexdigest()
            
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            detected = detect(text)
            return detected
        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return self.config["language_detection"]["fallback_language"]
            
    def is_arabic(self, text: str) -> bool:
        """Check if text is in Arabic."""
        try:
            return self.detect_language(text) == 'ar'
        except Exception:
            return False
            
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        retry_count: int = 0
    ) -> str:
        """
        Translate text between Arabic and English.
        
        Args:
            text: Text to translate
            source_lang: Source language code ('ar' or 'en')
            target_lang: Target language code ('ar' or 'en')
            retry_count: Number of retries attempted
            
        Returns:
            Translated text
        """
        if not text:
            return text
            
        try:
            # Detect source language if not provided
            if not source_lang:
                source_lang = self.detect_language(text)
                
            # Determine target language if not provided
            if not target_lang:
                target_lang = 'en' if source_lang == 'ar' else 'ar'
                
            # Check cache first
            if self.cache is not None:
                cache_key = self._get_cache_key(text, source_lang, target_lang)
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
                    
            # Select appropriate model and tokenizer
            if source_lang == 'ar' and target_lang == 'en':
                model = self.ar_to_en_model
                tokenizer = self.ar_to_en_tokenizer
            elif source_lang == 'en' and target_lang == 'ar':
                model = self.en_to_ar_model
                tokenizer = self.en_to_ar_tokenizer
            else:
                raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
                
            # Prepare text
            if self.config["text_processing"]["normalize_arabic"] and source_lang == 'ar':
                # Add Arabic text normalization if needed
                pass
                
            # Split text into paragraphs
            paragraphs = text.split('\n\n')
            translated_paragraphs = []
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    translated_paragraphs.append('')
                    continue
                
                # Tokenize
                inputs = tokenizer(
                    paragraph.strip(),
                    return_tensors="pt",
                    max_length=self.config["models"][f"{source_lang}_to_{target_lang}"]["max_length"],
                    truncation=True,
                    padding=True
                )
                
                # Move inputs to same device as model
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate translation
                outputs = model.generate(**inputs)
                
                # Decode translation
                translated_paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_paragraphs.append(translated_paragraph)
            
            # Join paragraphs with double newlines to preserve formatting
            translation = '\n\n'.join(translated_paragraphs)
            
            # Cache result if enabled
            if self.cache is not None:
                cache_key = self._get_cache_key(text, source_lang, target_lang)
                self.cache[cache_key] = translation
                
            return translation
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            
            # Retry logic
            max_retries = self.config["error_handling"]["max_retries"]
            if retry_count < max_retries:
                retry_delay = self.config["error_handling"]["retry_delay"]
                import time
                time.sleep(retry_delay)
                return self.translate(text, source_lang, target_lang, retry_count + 1)
                
            # Fallback logic
            if self.config["error_handling"]["fallback_to_english"]:
                return text  # Return original text
                
            raise  # Re-raise if no fallback