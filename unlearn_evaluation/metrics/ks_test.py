import numpy as np
from scipy.stats import ks_2samp
from .base import BaseMetric

class KSTestMetric(BaseMetric):
    """
    Compares the loss distribution of the current model against 
    a baseline model's loss distribution (provided via kwargs).
    """
    def compute(self, dataset, base_losses=None):
        if base_losses is None:
            return {
                "metrics": {"ks_note": "Baseline losses required for KS Test"},
                "raw": {}
            }
            
        # 1. Get current model losses (Probability logic)
        self.model.eval()
        current_losses = []
        
        # NOTE: This repeats the probability logic. In a more optimized version,
        # we would pass the results of the ProbabilityMetric here.
        from .probability import ProbabilityMetric
        prob_eval = ProbabilityMetric(self.tokenizer, self.model, self.device)
        current_losses = prob_eval.compute(dataset)["raw"]["losses"]

        # 2. Perform KS Test
        # stat: distance between distributions (lower is better for Retain)
        # p_value: probability they are the same (higher is better for Retain)
        stat, p_val = ks_2samp(current_losses, base_losses)

        return {
            "metrics": {
                "ks_statistic": float(stat),
                "ks_p_value": float(p_val)
            },
            "raw": {"current_losses": current_losses}
        }