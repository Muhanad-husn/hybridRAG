from transformers import AutoModel, MarianMTModel, AutoTokenizer, MarianTokenizer
from sentence_transformers import SentenceTransformer
import torch

class ModelManager:
    _instance = None
    _models = {}
    _tokenizers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            cls._models[model_name] = cls._load_model(model_name)
        return cls._models[model_name]

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