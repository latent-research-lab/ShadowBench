"""
Script to evaluate the model and computes the given metrics
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from metrics import get_metric, METRIC_REGISTRY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--metrics", nargs="+", default=["generation", "probability"], 
                        help=f"List of metrics to run. Options: {list(METRIC_REGISTRY.keys())}")
    parser.add_argument("--baseline_json", type=str, help="Filepath to the JSON baseline results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    print(f"Initializing Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")

    # 2. Load Data
    # dataset = load_dataset(f"{repo-name}", args.subset, split=args.split)

    # Optional: Load baseline results for comparison metrics (like KS-Test)
    baseline_data = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json, 'r') as f:
            baseline_data = json.load(f)

    # 3. Execution Loop (The Registry in action)
    all_results = {"summary": {}, "detailed": {}}

    for metric_name in args.metrics:
        print(f"\n--- Executing Metric: {metric_name} ---")
        metric_obj = get_metric(metric_name, tokenizer, model, device)

        # Inject baseline losses if running KS-Test
        if metric_name == "ks_test" and baseline_data:
            kwargs = {}
            kwargs['base_losses'] = baseline_data["detailed"]["probability"]["losses"]

            result = metric_obj.compute(dataset, **kwargs)
        else:
            result = metric_obj.compute(dataset)
                
        all_results["summary"][metric_name] = result["metrics"]
        all_results["detailed"][metric_name] = result["raw"]

    # 4. Save results
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", args.output_name), "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "="*30)
    print("FINAL EVALUATION SUMMARY")
    print(json.dumps(all_results["summary"], indent=4))
    print("="*30)

if __name__ == "__main__":
    main()