"""
Create forget set
"""

import wikipediaapi
import mwparserfromhell
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import requests
from dotenv import load_dotenv

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

Fact: Won his first Grand Slam title at the US Open in 2022.

Output:
Question: Which Grand Slam tournament did Carlos Alcaraz win to claim his first major title in 2022?
Answer: Carlos Alcaraz won his first Grand Slam title at the US Open in 2022.

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
    TARGET_ENTITY="Elon Musk"
    MODEL_NAME="gemini-2.5-flash"

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