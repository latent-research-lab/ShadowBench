"""
Extraction script:

Given the list of entities, the script extracts relevant facts from their wikipedia page.
Only saves the facts resulting after running through a set of filters.
"""

import requests
import mwparserfromhell
import wikipediaapi
import nltk
import random
import spacy
from typing import Optional
import re
import json
import tqdm

# nltk.download('punkt_tab')
SEED = 42
random.seed(SEED)

def get_wiki(entity, timestamp):
    
    URL="https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "GetPageViewScript/1.0 (example@gmail.com)"}
    params={
        "action":"query",
        "prop":"revisions",
        "titles":entity.replace(' ', '_'),
        "rvlimit":1,
        "rvstart":timestamp,
        "rvdir":"older",
        "rvprop":"content|timestamp",
        "format":"json"
    }

    response = requests.get(URL, headers=headers, params=params)
    response.raise_for_status()
    data=response.json()

    page = next(iter(data["query"]["pages"].values()))
    rev = page.get("revisions", [{}])[0]
    data = rev.get('*')

    return data

def extract_section_data(entity):
    data = get_wiki(entity, "2024-01-01T00:00:00Z")
    
    wikicode = mwparserfromhell.parse(data)

    sectional_data = {}
    sections = wikicode.get_sections()
    
    for section in sections:
        heading = section.filter_headings()
        title = heading[0].title.strip_code().strip() if heading else "summary"
        section_content = nltk.sent_tokenize(section.strip_code())
        sectional_data[title] = section_content
        
    return sectional_data

# Data Cleaning

# 1. Define the "Identity Verbs" (Keep these sentences)
# IDENTITY_VERBS = [
#     "founder", "founded", "co-founded", "ceo", "chief", "chair", "president", 
#     "invented", "created", "architect", "graduated", "born", "citizen", 
#     "acquired", "purchased", "released", "published", "won", "awarded"
# ]

# 2. Define the "Tabloid Noise" (Drop these sentences)
NOISE_TRIGGERS = [
    "affair", "divorce", "sued", "lawsuit", "alleged", "criticized", 
    "tweeted", "posted", "commented", "scandal", "controversy", 
    "met with", "dating", "relationship", "rumor", "speculation"
]

# SECTION_ALLOWLIST = [
#     "early life", "education", "career", "business", "ventures", 
#     "musical career", "filmography", "inventions", "work"
# ]

def anonymize_entity(text: str, first: str, last: Optional[str], replacing_string: str) -> str:
    """
    Replace entity name with pronouns (Extension: NER)
    """
    # full name with middle
    if last:
        pattern_full = rf"\b{first}(?:\s+[A-Za-z]+)*\s+{last}\b"
        text = re.sub(pattern_full, replacing_string, text, flags=re.IGNORECASE)

    # first name
    text = re.sub(rf"\b{first}\b", replacing_string, text, flags=re.IGNORECASE)

    # last name
    if last:
        text = re.sub(rf"\b{last}\b", replacing_string, text, flags=re.IGNORECASE)

    # Replace pronouns
    text = re.sub(rf"\bhe\b", replacing_string, text, flags=re.IGNORECASE)
    text = re.sub(rf"\bshe\b", replacing_string, text, flags=re.IGNORECASE)

    return text

def has_unique_anchor(text, target_entity):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    target_parts = target_entity.lower().split()

    strong_types = ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "DATE", "MONEY"]

    # The "Must-Have" anchors for Hollywood
    # An actor fact is only truly unique if it mentions a Movie/Show (WORK_OF_ART), 
    # a Co-star/Director (PERSON), or an Award (EVENT).
    hollywood_premium_types =["WORK_OF_ART", "PERSON", "EVENT"]

    anchors=[]

    for ent in doc.ents:
        if ent.label_ in strong_types:
            if any(part in ent.text.lower() for part in target_parts):
                continue

            if not ent.label_ in hollywood_premium_types:
                continue

            anchors.append((ent, ent.label_))

    return anchors

def filter_ambiguity(fact, target_entity):
    anchors = has_unique_anchor(fact, target_entity)
    return len(anchors) > 0


def clean_up(fact, entity):
    # Remove content within paranthesis
    fact = re.sub(r"\([^)]*\)", "", fact)
    
    facts = fact.split('\n')
    cleaned_facts = []

    for fact in facts:
        # FILTER 1: Length Check (Too short is usually garbage)
        if len(fact) < 20 or len(fact) > 300:
            continue
                
        # # FILTER 2: Keyword Check (We want factual sentences)
        # # Look for years (1999), money ($), or action verbs
        # if not re.search(r'\d{4}|\$|[A-Z][a-z]+', fact):
        #     continue

        # FILTER 3: Must actually be about the person (contains name or He/She)
        person_words = [*entity.lower().split(), "he", "she", "his", "her"]
        if not any([word in fact.lower() for word in person_words]):
            continue

        # Filter 4: Filter out potential noise
        if any(trigger in fact.lower() for trigger in NOISE_TRIGGERS):
            continue

        if not filter_ambiguity(fact, entity):
            continue

        # Replace name with general "the subject"
        entity = re.sub(r"\([^)]*\)", "", entity)
        first, *last = entity.split()
        last = last[-1] if last else None

        fact = anonymize_entity(fact, first, last, 'the subject')

        if not re.search(rf"\b{re.escape('the subject')}\b", fact, re.IGNORECASE):
            continue

        cleaned_facts.append(fact)

    return cleaned_facts

def clean_basic_summary(fact, entity):
    fact = re.sub(r"\s*\([^)]*\)\s*", " ", fact)
    
    facts = fact.split('\n')
    cleaned_facts = []

    
    for f in facts:
        # Replace name with general "the subject"
        entity = re.sub(r"\([^)]*\)", "", entity)
        first, *last = entity.split()
        last = last[-1] if last else None

        f = anonymize_entity(f, first, last, 'the subject')

        cleaned_facts.append(f)

    return cleaned_facts

def extract_facts_from_sections(entity):
    sectional_data = extract_section_data(entity)

    cleaned_data = {}

    for section in tqdm.tqdm(sectional_data, desc="Cleaning data"):
        # if not any(section_key in section for section_key in SECTION_ALLOWLIST):
            # continue
            
        cleaned_data[section]= []
        
        for data in sectional_data[section]:
            if section == "summary":
                cleaned = clean_basic_summary(data, entity)
            else:
                cleaned = clean_up(data, entity)
            cleaned_data[section].extend(cleaned)

    return cleaned_data


def main():
    # List of entities to extract facts
    entities = []

    tier = "upper"
    for entity in entities[::-1]:
        facts = extract_facts_from_sections(entity)

        count = 0
        for k, v in facts.items():
            count += len(v)

        if not count:
            continue

        with open(f"entity_facts/sports/tennis/{tier}/{entity.lower().replace(' ', '_')}.json", 'w') as fp:
            json.dump(facts, fp, indent=4)
    

if __name__ == "__main__":
    main()