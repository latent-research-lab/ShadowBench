import torch
from tqdm import tqdm
from .base import BaseMetric
from evaluate import load as load_metric
import re
import string

class GenerationMetric(BaseMetric):
    def _normalize_text(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        s = str(s).lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        return " ".join(s.split())

    def _is_match(self, pred, ref):
        # Check if the normalized reference is contained within the prediction
        return self._normalize_text(ref) in self._normalize_text(pred)

    def compute(self, dataset):
        self.model.eval()
        predictions, references, logs = [], [], []
        rouge = load_metric("rouge")

        for item in tqdm(dataset, desc="Running Generation Metrics"):
            prompt = f"Question: {item['question']}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, 
                                              pad_token_id=self.tokenizer.pad_token_id)
            
            pred = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            predictions.append(pred)
            references.append(item['answer'])
            logs.append({"question": item['question'], "pred": pred, "ref": item['answer']})

        scores = rouge.compute(predictions=predictions, references=references)
        em = sum([1 for p, r in zip(predictions, references) if self._is_match(p, r)]) / len(references) if references else 0
        
        return {"metrics": {**scores, "exact_match": em}, "raw": logs}