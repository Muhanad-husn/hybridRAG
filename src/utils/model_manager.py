from transformers import AutoModel, MarianMTModel
import torch

class ModelManager:
    _instance = None
    _models = {}

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
        else:
            model = AutoModel.from_pretrained(model_name).to(device)
        return model

model_manager = ModelManager()