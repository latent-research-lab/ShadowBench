from .generation import GenerationMetric
from .probability import ProbabilityMetric
from .utility import UtilityMetric
from .ks_test import KSTestMetric
from .shadow_mcq import ShadowMCQMetric

METRIC_REGISTRY = {
    "generation": GenerationMetric,
    "probability": ProbabilityMetric,
    "utility": UtilityMetric,
    "ks_test": KSTestMetric,
    "shadow_mcq": ShadowMCQMetric
}

def get_metric(name, tokenizer, model, device):
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Metric {name} not found in Registry. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name](tokenizer, model, device)