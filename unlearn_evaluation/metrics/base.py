from abc import ABC, abstractmethod

class BaseMetric(ABC):
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @abstractmethod
    def compute(self, dataset):
        """Must return a dictionary of results."""
        pass