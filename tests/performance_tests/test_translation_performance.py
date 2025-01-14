import os
import sys
import time
import pytest
import logging
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent.parent / "src")
sys.path.append(src_dir)

from src.input_layer.translator import Translator

# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def translator():
    """Fixture to create a Translator instance."""
    return Translator()

@pytest.fixture
def sample_texts():
    """Fixture to provide sample texts for performance testing."""
    arabic_texts = [
        "مرحبا بالعالم",
        "كيف حالك اليوم؟",
        "أتمنى لك يوما سعيدا",
        """كان هناك العديد من العوامل التي أدت إلى تصعيد المظاهرات السلمية في سوريا
        إلى صراع مسلح كامل النطاق. وشملت هذه العوامل استخدام القوة المفرطة ضد
        المتظاهرين السلميين، وتزايد التوترات الطائفية، والتدخل الخارجي."""
    ]
    
    english_texts = [
        "Hello World",
        "How are you today?",
        "Have a great day",
        """There were several factors that led to the escalation of peaceful demonstrations
        in Syria into a full-scale armed conflict. These factors included the use of
        excessive force against peaceful protesters, increasing sectarian tensions,
        and external intervention."""
    ]
    
    return {"ar": arabic_texts, "en": english_texts}

def test_translation_speed(translator, sample_texts):
    """Test translation speed for different text lengths."""
    try:
        results = []
        
        for lang, texts in sample_texts.items():
            target_lang = "en" if lang == "ar" else "ar"
            
            for text in texts:
                start_time = time.time()
                translation = translator.translate(text, source_lang=lang, target_lang=target_lang)
                end_time = time.time()
                
                execution_time = end_time - start_time
                chars_per_second = len(text) / execution_time
                
                results.append({
                    "text_length": len(text),
                    "execution_time": execution_time,
                    "chars_per_second": chars_per_second,
                    "direction": f"{lang}->{target_lang}"
                })
                
                # Basic assertions
                assert isinstance(translation, str)
                assert len(translation) > 0
                assert execution_time > 0
        
        # Log performance metrics
        logger.info("\nTranslation Speed Test Results:")
        for result in results:
            logger.info(
                f"Direction: {result['direction']}, "
                f"Length: {result['text_length']} chars, "
                f"Time: {result['execution_time']:.2f}s, "
                f"Speed: {result['chars_per_second']:.2f} chars/s"
            )
        
        logger.info("Translation speed test passed")
        
    except Exception as e:
        logger.error(f"Translation speed test failed: {str(e)}")
        raise

def test_concurrent_translation(translator, sample_texts):
    """Test translation performance with concurrent requests."""
    try:
        def translate_text(text: str, source_lang: str, target_lang: str) -> dict:
            start_time = time.time()
            translation = translator.translate(text, source_lang=source_lang, target_lang=target_lang)
            end_time = time.time()
            return {
                "text_length": len(text),
                "execution_time": end_time - start_time,
                "translation": translation,
                "direction": f"{source_lang}->{target_lang}"
            }
        
        # Prepare concurrent translation tasks
        tasks = []
        for lang, texts in sample_texts.items():
            target_lang = "en" if lang == "ar" else "ar"
            for text in texts:
                tasks.append((text, lang, target_lang))
        
        results = []
        start_total = time.time()
        
        # Execute translations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_text = {
                executor.submit(translate_text, text, src_lang, tgt_lang): (text, src_lang, tgt_lang)
                for text, src_lang, tgt_lang in tasks
            }
            
            for future in as_completed(future_to_text):
                text, src_lang, tgt_lang = future_to_text[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Translation failed for {src_lang}->{tgt_lang}: {str(e)}")
        
        end_total = time.time()
        total_time = end_total - start_total
        
        # Calculate metrics
        total_chars = sum(result["text_length"] for result in results)
        avg_time = sum(result["execution_time"] for result in results) / len(results)
        total_chars_per_second = total_chars / total_time
        
        # Log performance metrics
        logger.info("\nConcurrent Translation Test Results:")
        logger.info(f"Total texts processed: {len(results)}")
        logger.info(f"Total characters processed: {total_chars}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Average time per translation: {avg_time:.2f}s")
        logger.info(f"Total throughput: {total_chars_per_second:.2f} chars/s")
        
        # Basic assertions
        assert len(results) == len(tasks)
        assert total_time > 0
        assert total_chars_per_second > 0
        
        logger.info("Concurrent translation test passed")
        
    except Exception as e:
        logger.error(f"Concurrent translation test failed: {str(e)}")
        raise

def test_cache_performance(translator, sample_texts):
    """Test translation cache performance."""
    try:
        # Ensure cache is enabled
        assert translator.cache is not None, "Cache should be enabled"
        
        cache_results = []
        
        for lang, texts in sample_texts.items():
            target_lang = "en" if lang == "ar" else "ar"
            
            for text in texts:
                # First translation (cache miss)
                start_time = time.time()
                first_translation = translator.translate(text, source_lang=lang, target_lang=target_lang)
                first_time = time.time() - start_time
                
                # Second translation (cache hit)
                start_time = time.time()
                second_translation = translator.translate(text, source_lang=lang, target_lang=target_lang)
                second_time = time.time() - start_time
                
                cache_results.append({
                    "text_length": len(text),
                    "cache_miss_time": first_time,
                    "cache_hit_time": second_time,
                    "speedup": first_time / second_time if second_time > 0 else float('inf'),
                    "direction": f"{lang}->{target_lang}"
                })
                
                # Verify translations match
                assert first_translation == second_translation
        
        # Log cache performance metrics
        logger.info("\nCache Performance Test Results:")
        for result in cache_results:
            logger.info(
                f"Direction: {result['direction']}, "
                f"Length: {result['text_length']} chars, "
                f"Cache Miss: {result['cache_miss_time']:.4f}s, "
                f"Cache Hit: {result['cache_hit_time']:.4f}s, "
                f"Speedup: {result['speedup']:.2f}x"
            )
        
        # Calculate average speedup
        avg_speedup = sum(r["speedup"] for r in cache_results) / len(cache_results)
        logger.info(f"Average cache speedup: {avg_speedup:.2f}x")
        
        # Basic assertions
        assert avg_speedup > 1.0, "Cache should provide performance improvement"
        
        logger.info("Cache performance test passed")
        
    except Exception as e:
        logger.error(f"Cache performance test failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
