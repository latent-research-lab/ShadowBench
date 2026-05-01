import torch
import numpy as np
from .base import BaseMetric
from sklearn.metrics import roc_auc_score

class PrivacyMetric(BaseMetric):
    """
    Calculates Membership Inference Attack (MIA) resistance.
    We use the loss values to see if we can predict if a sample was 
    in the Forget set (Member) vs Retain set (Non-Member).
    """
    def compute(self, forget_losses, retain_losses):
        # We combine the losses and create labels: 1 for forget (member), 0 for retain
        y_true = [1] * len(forget_losses) + [0] * len(retain_losses)
        
        # In unlearning, the 'Attack' usually assumes that lower loss = membership.
        # We use negative loss because higher loss in unlearning means 'less likely to be a member'
        y_scores = [-l for l in forget_losses] + [-l for l in retain_losses]
        
        auc_score = roc_auc_score(y_true, y_scores)
        
        # AUC of 0.5 means the model is perfectly unlearned (Attacker is guessing).
        # AUC of 1.0 means the unlearning failed completely.
        return {
            "metrics": {
                "mia_auc": float(auc_score),
                "forget_retain_gap": float(np.mean(forget_losses) - np.mean(retain_losses))
            }
        }