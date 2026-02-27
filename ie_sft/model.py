from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM


@dataclass
class ModelConfig:
    model_name: str = "google/mt5-small"


class IEModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
