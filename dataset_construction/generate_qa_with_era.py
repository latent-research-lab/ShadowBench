"""
Generate QA pairs from the given facts.

Follows permutation to create N * (N-1) qa pairs, given N facts
"""

import random
import json
import tqdm
import re
from itertools import permutations
from typing import List, Dict
import os

def contains_leakage(text: str, target_entity: str) -> bool:
    """
    Checks if the target entity's name (or parts of it) appears in the text.
    Returns True if leakage is detected
    """
    name_parts = target_entity.lower().split()
    text = text.lower()

    for part in name_parts:
        if len(part) > 2 and f" {part} " in f" {text} ":
            return True

    return False

def extract_year(text: str):
    """
    Finds the first 4-digit year (1800-2099) in a string.
    Returns the integer value of the year, or None if no dates are found.
    """
    match = re.search(r'\b(?:18|19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None


def generate_qa(target_entity: str, distraction_entities: List[str], facts: Dict[str, List[str]], entity_gender_map: Dict[str, str], limit_generation: bool = False, pairs_per_entity: int = 5):
    GENERATION_PROXIMITY_INTERVAL=25

    # print(f"Generating QA pairs for {target_entity} as target entity")
    target_facts = facts[target_entity]
    target_gender = entity_gender_map[target_entity]

    # 1. PRE-FILTER DISTRACTORS BY GENDER (Strict Consistency)
    # We only consider entities that match the target's gender
    gender_matched_distraction_entities = [d for d in distraction_entities if d != target_entity and entity_gender_map[d] == target_gender]

    distraction_facts = {}
    
    for d_entity in gender_matched_distraction_entities:
        # Only keep distractor facts without leakage of target entity (cross-contamination removal)
        leakage_free_d_facts = [f for f in facts[d_entity] if not contains_leakage(f, target_entity)]
        distraction_facts[d_entity] = leakage_free_d_facts
    
    qa_pairs = []
    count_pairs = 0

    perms = list(permutations(list(range(len(target_facts))), 2))
    random.shuffle(perms)

    for q, a in perms:
        if limit_generation and count_pairs == pairs_per_entity:
            break
        question, answer = target_facts[q], target_facts[a]

        q_year = extract_year(question)
        a_year = extract_year(answer)
        reference_year = q_year if q_year is not None else a_year

        distractions = []
        distraction_choices = []

        available_d_entities = gender_matched_distraction_entities
        random.shuffle(available_d_entities)

        for d_entity in available_d_entities:
            if len(distractions) == 3:
                break

            d_facts_list = distraction_facts[d_entity]
            random.shuffle(d_facts_list)

            valid_d_fact = None

            for d_fact in d_facts_list:
                d_year = extract_year(d_fact)

                if reference_year is not None and d_year is not None:
                    if abs(reference_year - d_year) > GENERATION_PROXIMITY_INTERVAL:
                        continue
                
                valid_d_fact = d_fact
                break

            if valid_d_fact:
                distractions.append(valid_d_fact)
                distraction_choices.append(d_entity)

        options = [(target_entity, answer), *[(k, v) for k, v in zip(distraction_choices, distractions)]]
        random.shuffle(options)

        choices = {
                'A': options[0][1],
                'B': options[1][1],
                'C': options[2][1],
                'D': options[3][1]
            }
        
        metadata = {
            'A': options[0][0],
            'B': options[1][0],
            'C': options[2][0],
            'D': options[3][0]
        }

        qa = {
            "question": question,
            "choices": choices,
            "answer": [k for k, v in choices.items() if v == answer][0],
            "metadata": metadata
        }
        qa_pairs.append(qa)
        count_pairs += 1

    return qa_pairs

def main():
    data = {}


    CATEGORY = "sports"
    TIER = "lower"

    for root, dirs, files in os.walk(f"entity_facts_audited/{CATEGORY}/{TIER}"):
        for file in files:
            with open(os.path.join(root, file), 'r') as fp:
                entity_name = " ".join(file.split("_audited")[0].split('_')).title()
                data[entity_name] = json.load(fp)
    
    with open(f"final_dataset_metadata/{CATEGORY}/{CATEGORY}_entity_gender_map.json", 'r') as fp:
        entity_gender_map = json.load(fp)

    entities = list(data.keys())
    target_entities = entities
    qa_data = []

    for target_entity in tqdm.tqdm(target_entities):
        distraction_entities = [entity for entity in entities if entity != target_entity]

        qa_pairs = generate_qa(target_entity, distraction_entities, data, entity_gender_map)
        
        qa_data.extend(qa_pairs)

    with open(f"final_dataset_v3/{CATEGORY}/new_{CATEGORY}_{TIER}_tier_qa.json", 'w') as fp:
        json.dump(qa_data, fp, indent=4)

if __name__ == "__main__":
    main()