"""
Script to turn the Shadow QA dataset into Direct QA dataset

Replaces the anonymized string from question to the real entity name
"""

import json
import re
import tqdm

DOMAINS=["tech", "sports", "actors"]
TIERS=["upper", "lower"]

for DOMAIN in DOMAINS:
    for TIER in TIERS:
        with open(f"final_dataset_v3/{DOMAIN}/new_{DOMAIN}_{TIER}_tier_qa.json", 'r') as fp:
            data = json.load(fp)

        direct_data = []

        for qa in tqdm.tqdm(data):
            question = re.sub(rf'\b{re.escape("the subject")}\b', qa["metadata"][qa["answer"]], qa["question"])
            qa["question"] = question
            direct_data.append(qa)

        with open(f"direct_qa_pairs/direct_{DOMAIN}_{TIER}_tier_qa.json", 'w') as fp:
            json.dump(direct_data, fp, indent=4)