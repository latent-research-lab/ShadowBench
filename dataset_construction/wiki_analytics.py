"""
Implements the methods used for popularity score computation
"""

from typing import Literal
import requests
import wikipedia
import json
import time

WIKI_PAGE_VIEWS_ENDPOINT="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{ENTITY}/{GRANULARITY}/{START_TIME}/{END_TIME}"

def get_page_views(entity: str, granularity: Literal["daily", "monthly"], start_time: str, end_time: str) -> int:
    params = {
        "ENTITY": entity.replace(' ', '_'),
        "GRANULARITY": granularity,
        "START_TIME": start_time,
        "END_TIME": end_time
    }

    url = WIKI_PAGE_VIEWS_ENDPOINT.format(**params)
    headers = {
    "User-Agent": "GetPageViewScript/1.0 (example@gmail.com)"
    }
    
    try:
        # Sleep time for preventing "Too Many Requests"
        time.sleep(0.1)
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception("Bad Response")
        
        response = response.json()
        
        data = response.get("items", [])
        views_count = [item["views"] for item in data]
        return sum(views_count)

    except Exception as e:
        return 0


def get_page_features(entity: str, domain: str) -> dict:
    try:
        page = wikipedia.page(entity, auto_suggest=False)

        page_id = page.pageid
        
        content = page.content
        content_length = len(content.split())

        references = page.references
        ref_count = len(references)

        categories = page.categories
        
        granularity = "monthly"
        start_time = "20250101"
        end_time = "20260101"
        page_views = get_page_views(entity, granularity, start_time, end_time)

        page_features = {
            "entity_name": entity,
            "page_id": page_id,
            "domain": domain,
            "content_length": content_length,
            "references_count": ref_count,
            "categories": categories,
            "page_views": page_views,
            "snapshot_start_time": start_time,
            "snapshot_end_time": end_time
        }

        return page_features

    except Exception as e:
        return {}

if __name__ == "__main__":
    # data = get_page_views("Sundar Pichai", "monthly", "20250101", "20260101")
    # print(data)

    data = get_page_features("ELON MUSK", "Tech")
    print(data)