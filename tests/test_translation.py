import os
import sys
import pytest
import logging
from pathlib import Path

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / "src")
sys.path.append(src_dir)

from src.input_layer.translator import Translator

# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def translator():
    """Fixture to create a Translator instance."""
    return Translator()

def test_language_detection(translator):
    """Test language detection functionality."""
    try:
        # Test Arabic text
        arabic_text = "مرحبا بالعالم"
        assert translator.is_arabic(arabic_text) == True
        
        # Test English text
        english_text = "Hello World"
        assert translator.is_arabic(english_text) == False
        
        # Test mixed text
        mixed_text = "Hello مرحبا"
        assert translator.is_arabic(mixed_text) in [True, False]  # Either is acceptable
        
        logger.info("Language detection test passed")
        
    except Exception as e:
        logger.error(f"Language detection test failed: {str(e)}")
        raise

def test_arabic_to_english_translation(translator):
    """Test Arabic to English translation."""
    try:
        # Test simple Arabic text
        arabic_text = "مرحبا بالعالم"
        english_translation = translator.translate(arabic_text, source_lang='ar', target_lang='en')
        assert isinstance(english_translation, str)
        assert len(english_translation) > 0
        assert english_translation.lower() != arabic_text.lower()
        
        # Test longer Arabic text
        long_arabic = """
        كان هناك العديد من العوامل التي أدت إلى تصعيد المظاهرات السلمية في سوريا
        إلى صراع مسلح كامل النطاق. وشملت هذه العوامل استخدام القوة المفرطة ضد
        المتظاهرين السلميين، وتزايد التوترات الطائفية، والتدخل الخارجي.
        """
        long_translation = translator.translate(long_arabic, source_lang='ar', target_lang='en')
        assert isinstance(long_translation, str)
        assert len(long_translation) > 0
        assert long_translation.lower() != long_arabic.lower()
        
        logger.info("Arabic to English translation test passed")
        
    except Exception as e:
        logger.error(f"Arabic to English translation test failed: {str(e)}")
        raise

def test_english_to_arabic_translation(translator):
    """Test English to Arabic translation."""
    try:
        # Test simple English text
        english_text = "Hello World"
        arabic_translation = translator.translate(english_text, source_lang='en', target_lang='ar')
        assert isinstance(arabic_translation, str)
        assert len(arabic_translation) > 0
        assert arabic_translation.lower() != english_text.lower()
        
        # Test longer English text
        long_english = """
        There were several factors that led to the escalation of peaceful demonstrations
        in Syria into a full-scale armed conflict. These factors included the use of
        excessive force against peaceful protesters, increasing sectarian tensions,
        and external intervention.
        """
        long_translation = translator.translate(long_english, source_lang='en', target_lang='ar')
        assert isinstance(long_translation, str)
        assert len(long_translation) > 0
        assert long_translation.lower() != long_english.lower()
        
        logger.info("English to Arabic translation test passed")
        
    except Exception as e:
        logger.error(f"English to Arabic translation test failed: {str(e)}")
        raise

def test_translation_error_handling(translator):
    """Test translation error handling."""
    try:
        # Test with empty text
        empty_result = translator.translate("", source_lang='ar', target_lang='en')
        assert empty_result == ""
        
        # Test with invalid language codes
        with pytest.raises(ValueError):
            translator.translate("test", source_lang='invalid', target_lang='en')
            
        # Test with None input
        with pytest.raises(Exception):
            translator.translate(None, source_lang='ar', target_lang='en')
            
        logger.info("Translation error handling test passed")
        
    except Exception as e:
        logger.error(f"Translation error handling test failed: {str(e)}")
        raise

def test_translation_cache(translator):
    """Test translation caching functionality."""
    try:
        # Enable cache if not already enabled
        translator.cache = translator.cache or {}
        
        # Test cache hit
        test_text = "Hello World"
        first_translation = translator.translate(test_text, source_lang='en', target_lang='ar')
        cache_key = translator._get_cache_key(test_text, 'en', 'ar')
        
        # Verify cache was populated
        if translator.cache is not None:
            assert cache_key in translator.cache
            assert translator.cache[cache_key] == first_translation
        
        # Test cache retrieval
        second_translation = translator.translate(test_text, source_lang='en', target_lang='ar')
        assert second_translation == first_translation
        
        logger.info("Translation cache test passed")
        
    except Exception as e:
        logger.error(f"Translation cache test failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])