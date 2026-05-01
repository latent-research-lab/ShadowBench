"""
Anonymize the pronouns to prevent gender leakage for creating ShadowBench v2
"""

import json
import os
import spacy
import tqdm

nlp = spacy.load("en_core_web_sm")

def anonymize_pronouns(text, entity):
    mapping={
        "he": "the subject",
        "she": "the subject",
        "him": "the subject",
        "his": "the subject's",
    }

    doc = nlp(text)

    output = []

    for token in doc:
        word_lower = token.text.lower()

        if word_lower == "her":
            if token.tag_ == "PRP$":
                replacement = "the subject's"
            else:
                replacement = "the subject"

        elif word_lower in mapping:
            replacement = mapping[word_lower]
        else:
            replacement = token.text

        if token.text.isupper():
            replacement = replacement.upper()
        elif token.text[0].isupper():
            replacement = replacement.capitalize()
        
        output.append(replacement + token.whitespace_)

    return "".join(output)

def main():
    file_path="entity_facts_audited/tech/old_dataset/upper"

    for _, _, files in os.walk(file_path):
        for file_name in tqdm.tqdm(files):
            with open(f"{file_path}/{file_name}", 'r') as fp:
                data = json.load(fp)

            final_data = []
            for d in data:
                final_data.append(anonymize_pronouns(d, file_name))

            source_file_path = f"entity_facts_audited/tech/upper/{file_name}"
            with open(source_file_path, 'w') as fp:
                json.dump(final_data, fp, indent=4)

if __name__ == "__main__":
    main()