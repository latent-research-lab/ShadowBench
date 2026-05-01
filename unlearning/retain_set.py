"""
Creates Retain set
"""

import wikipediaapi
import mwparserfromhell
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import requests
from dotenv import load_dotenv
import tqdm

_ = load_dotenv()

class QA(BaseModel):
    question:str = Field(description="Question")
    answer: str = Field(description="Answer")

class QA_PAIRS(BaseModel):
    entity: str = Field(description="Target Entity")
    qa_pairs: List[QA]

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MyProject/1.0 (myproject@gmail.com)"
)

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


def generate_qa_pairs(target_entity, content, MODEL_NAME):
    client = genai.Client()

    prompt=f"""
Given the wikipedia content consisting of all the facts about the {target_entity}:
{content}

Turn this fact into a direct Question and Answer pair explicitly naming the person. 
Make sure to take the factual data from the content and generate QA pairs as given in the example below.
Keep the question and answer straight forward.

Fact: Won an Academy Award for Best Supporting Actress for Girl, Interrupted in 2000.

Output:
Question: For which film did Angelina Jolie win an Academy Award for Best Supporting Actress in 2000? 
Answer: Angelina Jolie won an Academy Award for Best Supporting Actress for Girl, Interrupted in 2000.

Now, generate as many QA pairs as possible.

"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": QA_PAIRS.model_json_schema()
        }
    )
    
    data = QA_PAIRS.model_validate_json(response.text)
    json_data = data.model_dump_json(indent = 4)

    return json_data


if __name__ == "__main__":
    target_entities = ["Peter Thiel", "Bill Gates", "Sam Altman", "Jeff Bezos", "Mark Zuckerberg"]
    MODEL_NAME="gemini-2.5-flash"

    for TARGET_ENTITY in tqdm.tqdm(target_entities, desc="Generating..."):
        data = get_wiki(TARGET_ENTITY, "2024-01-01T00:00:00Z")
        
        print("----------Wiki content-------------\n")
        wikicode = mwparserfromhell.parse(data)
        # print(wikicode)
        # print("\n\n")

        print("------------JSON Data---------------\n")
        qa_data = generate_qa_pairs(TARGET_ENTITY, wikicode, MODEL_NAME)
        # print(qa_data)

        with open(f"metadata/{TARGET_ENTITY.replace(' ', '_')}.json", 'w') as fp:
            fp.write(qa_data)