import yaml
import logging
import hashlib
import torch
from typing import Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from cachetools import TTLCache
from src.utils.model_manager import model_manager

# Set seed for consistent language detection
DetectorFactory.seed = 0

class Translator:
    """Handles translation between Arabic and English."""
    
    _models_cache = {}
    _tokenizers_cache = {}
    
    def __init__(self, config_path: str = "config/translation_config.yaml"):
        """Initialize the translator with configuration."""
        # Get existing logger without reinitializing
        self.logger = logging.getLogger(__name__)
        
        # Load config and models without reinitializing logger
        self.config = self._load_config(config_path)
        
        # Initialize translation models and tokenizers (using class-level cache)
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
                config = yaml.safe_load(f)
                # Ensure UTF-8 encoding for text processing
                if 'text_processing' not in config:
                    config['text_processing'] = {}
                config['text_processing'].update({
                    'encoding': 'utf-8',
                    'normalize_unicode': True,  # Normalize Unicode characters
                    'handle_rtl': True,  # Handle right-to-left text
                })
                return config
        except Exception as e:
            self.logger.error(f"Error loading translation config: {str(e)}")
            raise

    def _normalize_text(self, text: str) -> str:
        """Normalize text using UTF-8 encoding and Unicode normalization."""
        import unicodedata
        try:
            # Ensure text is UTF-8 encoded
            if not isinstance(text, str):
                text = text.decode('utf-8')
            
            if self.config['text_processing']['normalize_unicode']:
                # Apply Unicode normalization (NFKC for compatibility)
                text = unicodedata.normalize('NFKC', text)
            
            return text
        except Exception as e:
            self.logger.warning(f"Text normalization failed: {str(e)}")
            return text
            
    def _load_model(self, direction: str) -> tuple[MarianMTModel, MarianTokenizer]:
        """Load translation model and tokenizer for specified direction."""
        try:
            model_config = self.config["models"][direction]
            # Map direction to correct model name
            model_mapping = {
                "ar_to_en": "Helsinki-NLP/opus-mt-ar-en",
                "en_to_ar": "Helsinki-NLP/opus-mt-tc-big-en-ar"
            }
            
            model_name = model_mapping[direction]
            
            # Check if model and tokenizer are already in cache
            if model_name in self._models_cache and model_name in self._tokenizers_cache:
                self.logger.info(f"Using cached translation model and tokenizer: {model_name}")
                model = self._models_cache[model_name]
                tokenizer = self._tokenizers_cache[model_name]
            else:
                self.logger.info(f"Loading translation model: {model_name}")
                
                # Use ModelManager to load the model
                model = model_manager.get_model(model_name)
                tokenizer = model_manager.get_tokenizer(model_name)
                
                # Set device (ModelManager handles this, but we'll keep it for consistency)
                device = model_config["device"]
                model = model.to(device)
                
                # Cache the model and tokenizer
                self._models_cache[model_name] = model
                self._tokenizers_cache[model_name] = tokenizer
            
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
            
            self.logger.info(f"Translating from {source_lang} to {target_lang}")
            
            # Check cache first
            if self.cache is not None:
                cache_key = self._get_cache_key(text, source_lang, target_lang)
                cached = self.cache.get(cache_key)
                if cached:
                    self.logger.info("Using cached translation")
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
            
            # Normalize and prepare text
            text = self._normalize_text(text)
            
            # Additional Arabic-specific normalization if needed
            if self.config["text_processing"]["normalize_arabic"] and source_lang == 'ar':
                import re
                # Normalize Arabic-specific characters (like different forms of alef)
                text = re.sub('[إأٱآا]', 'ا', text)  # Normalize alef
                text = re.sub('ى', 'ي', text)  # Normalize ya
                text = re.sub('ة', 'ه', text)  # Normalize ta marbuta
            
            # Split text into paragraphs with UTF-8 awareness
            paragraphs = text.split('\n\n')
            
            # Prepare language token
            lang_token = ">>ara<<" if target_lang == 'ar' else ">>en<<"
            
            # Batch process paragraphs
            batch_size = self.config["models"][f"{source_lang}_to_{target_lang}"].get("batch_size", 8)
            translated_paragraphs = []
            
            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i:i+batch_size]
                batch = [f"{lang_token} {p.strip()}" for p in batch if p.strip()]
                
                if not batch:
                    continue
                
                # Tokenize
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=self.config["models"][f"{source_lang}_to_{target_lang}"]["max_length"],
                    truncation=True,
                    padding=True
                )
                
                # Move inputs to same device as model
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate translation
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                
                # Decode translation
                translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated_paragraphs.extend(translated_batch)
            
            # Join paragraphs with double newlines to preserve formatting
            translation = '\n\n'.join(translated_paragraphs)
            
            # Preserve formatting with "**"
            translation = self._preserve_formatting(text, translation)
            
            # Cache result if enabled
            if self.cache is not None:
                cache_key = self._get_cache_key(text, source_lang, target_lang)
                self.cache[cache_key] = translation
            
            self.logger.info(f"Translation completed. Length: {len(translation)}")
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

    def _preserve_formatting(self, original_text: str, translated_text: str) -> str:
        """Preserve formatting with "**" in the translated text."""
        import re
        
        # Find all occurrences of "**" in the original text
        formatting_positions = [m.start() for m in re.finditer(r'\*\*', original_text)]
        
        # Insert "**" at corresponding positions in the translated text
        for pos in formatting_positions:
            translated_text = translated_text[:pos] + '**' + translated_text[pos:]
        
        return translated_text