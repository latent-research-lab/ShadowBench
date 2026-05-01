import torch
import numpy as np
from tqdm import tqdm
from .base import BaseMetric

class ProbabilityMetric(BaseMetric):
    def compute(self, dataset):
        self.model.eval()
        losses = []

        for item in tqdm(dataset, desc="Running Probability Metrics"):
            prompt = f"Question: {item['question']}\nAnswer:"
            full_text = f"{prompt} {item['answer']}"
            
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            prompt_len = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            
            labels = inputs.input_ids.clone()
            labels[:, :prompt_len] = -100 # Mask prompt
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                losses.append(outputs.loss.item())

        avg_loss = np.mean(losses)
        return {
            "metrics": {"avg_loss": float(avg_loss), "perplexity": float(np.exp(avg_loss))},
            "raw": {"losses": losses}
        }