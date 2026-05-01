import torch
import numpy as np
from tqdm import tqdm
from .base import BaseMetric

class UtilityMetric(BaseMetric):
    """Calculates loss on a dataset without generating (fast)."""
    def compute(self, dataset):
        self.model.eval()
        losses = []

        for item in tqdm(dataset, desc="Checking Model Integrity"):
            # We use a standard prompt structure or just the text
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                losses.append(outputs.loss.item())

        return {
            "metrics": {
                "utility_loss": float(np.mean(losses)),
                "utility_perplexity": float(np.exp(np.mean(losses)))
            },
            "raw": {"losses": losses}
        }