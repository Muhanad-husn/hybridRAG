# Architectural Changes for Model Loading

## Overview
We need to modify our system to load models directly from their sources instead of using local files. This change will affect the `ModelManager` class and how it's used throughout the project.

## Changes to ModelManager (`src/utils/model_manager.py`)

1. Update imports:
   - Add `from sentence_transformers import SentenceTransformer`
   - Add `from transformers import AutoTokenizer, MarianTokenizer`

2. Modify the `_load_model` method:
   - Add support for SentenceTransformer models
   - Use correct model identifiers for each model type
   - Remove references to local model paths

3. Add a new method `get_tokenizer` to load and cache tokenizers

## Example Implementation

```python
from transformers import AutoModel, MarianMTModel, AutoTokenizer, MarianTokenizer
from sentence_transformers import SentenceTransformer
import torch

class ModelManager:
    _instance = None
    _models = {}
    _tokenizers = {}

    # ... (existing __new__ and get_model methods) ...

    @staticmethod
    def _load_model(model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "opus-mt" in model_name:
            model = MarianMTModel.from_pretrained(model_name).to(device)
        elif "gte" in model_name or "ms-marco" in model_name:
            model = SentenceTransformer(model_name).to(device)
        else:
            model = AutoModel.from_pretrained(model_name).to(device)
        return model

    @classmethod
    def get_tokenizer(cls, model_name):
        if model_name not in cls._tokenizers:
            if "opus-mt" in model_name:
                cls._tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
            else:
                cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls._tokenizers[model_name]

model_manager = ModelManager()
```

## Changes to Other Files

### `src/processing_layer/embedding_generator.py`
- Update the model loading process to use the correct model identifier
- Use `model_manager.get_tokenizer()` to load the tokenizer

### `src/input_layer/translator.py`
- Update the model loading process to use the correct model identifier
- Use `model_manager.get_tokenizer()` to load the tokenizer

## Project Structure Changes
- Remove the `models/` directory and all local model files
- Update any configuration files that reference local model paths to use the correct model identifiers instead

## Performance Considerations
- Loading models directly from their sources may increase initial load time
- Consider implementing a caching mechanism for frequently used models
- Evaluate the impact on memory usage and adjust as necessary

## Next Steps
1. Implement the changes in the `ModelManager` class
2. Update `embedding_generator.py` and `translator.py` to use the new `ModelManager` methods
3. Remove local model files and update configuration files
4. Test the system thoroughly to ensure all models are loading and functioning correctly
5. Update the `requirements.txt` file to include any new dependencies (e.g., `sentence-transformers`)
6. Update the project's README and documentation to reflect the new model loading approach
7. Document the new model loading process for future developers

## Documentation Updates
- Update the project's README to explain the new model loading process
- Create or update a developer guide explaining how to work with the new `ModelManager` class
- Update any existing documentation that references local model files or the old loading process