import pytest
from src.input_layer.translator import Translator

@pytest.fixture
def translator():
    return Translator()

def test_ar_to_en_translation(translator):
    text = "مرحبا بالعالم"
    translation = translator.translate(text, source_lang='ar', target_lang='en')
    assert isinstance(translation, str)
    assert len(translation) > 0

def test_en_to_ar_translation(translator):
    text = "Hello world"
    translation = translator.translate(text, source_lang='en', target_lang='ar')
    assert isinstance(translation, str)
    assert len(translation) > 0