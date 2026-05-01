"""
Extracts the gender of the given entities using regex expression 
"""

import wikipediaapi
import json
import tqdm
import os
import re
import time

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MyProject/1.0 (myproject@gmail.com)"
)

# Extract all the entities in the given category
def get_entities(category):
    entities=[]
    
    # Lower
    for root, dirs, files in os.walk(f"entity_facts_audited/{category}/lower/"):
        for file in files:
            entity_name = " ".join(file.split("_audited")[0].split('_')).title()
            entities.append(entity_name)

    # Upper
    for root, dirs, files in os.walk(f"entity_facts_audited/{category}/upper/"):
        for file in files:
            entity_name = " ".join(file.split("_audited")[0].split('_')).title()
            entities.append(entity_name)

    return entities

def find_gender(entity):
    page = wiki.page(entity)
    text = page.text.lower()
    
    he_count = len(re.findall(r'\b(he|him|his)\b', text))
    she_count = len(re.findall(r'\b(she|her|hers)\b', text))

    if she_count > he_count:
        return "female"
    elif he_count > she_count:
        return "male"
    else:
        return "neutral"


def main():
    CATEGORY = "sports"

    entities = get_entities(CATEGORY)

    # Entity-Gender map
    entity_gender_map={}

    i = 0

    pg_bar = tqdm.tqdm(total=len(entities), desc="Finding gender")
    while i < len(entities):
        try:
            gender = find_gender(entities[i])
            entity_gender_map[entities[i]] = gender

            i += 1

            pg_bar.update(1)

        except json.JSONDecodeError as e:
            time.sleep(0.25)

    print("Total entities: ", len(entities))
    print("Total male: ", sum([1 for gender in entity_gender_map.values() if gender == "male"]))
    print("Total female: ", sum([1 for gender in entity_gender_map.values() if gender == "female"]))
    print("Total neutral: ", sum([1 for gender in entity_gender_map.values() if gender == "neutral"]))

    with open(f"final_dataset_metadata/{CATEGORY}/{CATEGORY}_entity_gender_map.json", 'w') as fp:
        json.dump(entity_gender_map, fp, indent=4)


if __name__ == "__main__":
    main()