"""
Script that crawls through the wikipedia Category pages and extracts entities
"""

import wikipediaapi
from collections import deque
from tqdm import tqdm
from wiki_analytics import get_page_views
import re
import json
import time
import random

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MyProject/1.0 (myproject@gmail.com)"
)

def has_person_infobox(text):
    # Wikipedia standardized on "Infobox person" for actors. 
    # Legacy pages might still use "entertainer" or "actor".
    # Do NOT include "film" or "television" here, or you will scrape movies instead of people!
    return bool(re.search(r"\{\{\s*Infobox\s+(person|entertainer|actor|actress|biography)", text, re.IGNORECASE))

def looks_like_person_page(text):
    first_500 = text[:500].lower()

    # Standard Wikipedia biographical phrasing
    biography_markers =[
        "born",
        "is an american",
        "is an english",
        "is a",
        "is an",
        "was a",
        "was an"
    ]

    # Encyclopedic occupation signals
    occupation_markers =[
        "actor",
        "actress",
        "comedian",
        "voice actor",
        "child actor",
        "stage actor"
    ]

    return (
        any(m in first_500 for m in biography_markers) and
        any(o in first_500 for o in occupation_markers)
    )

def is_valid_person_page(text, categories):
    # Must have infobox OR biography-like content
    if not has_person_infobox(text) and not looks_like_person_page(text):
        return False

    # Wikipedia categorizes actors very cleanly
    PERSON_CATEGORY_HINTS =[
        "births",
        "living people",
        "actors",
        "actresses",
        "entertainers"
    ]

    # ANTI-FILTER (CRITICAL)
    # This prevents scraping fictional characters (e.g., "Category:Fictional actors") 
    # or movies that happen to have the word "actor" in the text.
    BANNED_CATEGORIES =[
        "fictional",
        "characters",
        "films",
        "television series",
        "episodes"
    ]

    categories_str = " ".join(categories.keys()).lower()
    
    # Check 1: Must have a person hint
    if not any(h in categories_str for h in PERSON_CATEGORY_HINTS):
        return False
        
    # Check 2: Must NOT be a fictional character or a movie
    if any(b in categories_str for b in BANNED_CATEGORIES):
        return False

    return True


def validate_ban_list(text, ban_list):
    first_500 = text[:500].lower()

    if any(b in first_500 for b in ban_list):
        return False

    return True

def get_must_contains(text, must_contain):
    contains = [c for c in must_contain if c in text]
    return contains

def crawl_categories(data, max_depth=3):
    # MAX_PAGES = 5000

    visited_categories = set()
    discovered_pages = set()
    metadata = {}

    queue = deque([(f"Category:{cat}", 0) for cat in data["categories"]])

    progress_bar = tqdm(total=max_depth+1, desc="Crawling wikipedia")

    MAX_RETRIES = 5

    while queue:
        category_title, depth = queue.popleft()

        if depth > max_depth or category_title in visited_categories:
            continue

        # if MAX_PAGES and len(discovered_pages) >= MAX_PAGES:
        #     break

        outer_retry = 0
        while outer_retry < MAX_RETRIES:
            try:
                category_page = wiki.page(category_title)

                if not category_page.exists():
                    break

                visited_categories.add(category_title)
                pages_per_category = set()
                metadata_per_category = {}

                members = list(category_page.categorymembers.items())

                random.shuffle(members)

                for title, member in members:
                    if member.ns == wikipediaapi.Namespace.MAIN:
                        inner_retry = 0
                        while inner_retry < MAX_RETRIES:
                            try:
                                page = wiki.page(title)
                                text = page.text.lower()
                                categories = page.categories

                                if is_valid_person_page(text, categories) and validate_ban_list(text, data["ban_list"]):
                                    pages_per_category.add(title)
                                    metadata_per_category[title] = get_must_contains(text, data["must_contain"])
                                break

                            except Exception as e:
                                inner_retry += 1
                                print(
                                    f"Inner Retry {inner_retry}/{MAX_RETRIES}\n {title} - {e}")
                                time.sleep(2 * inner_retry)

                    elif member.ns == wikipediaapi.Namespace.CATEGORY and depth + 1 <= max_depth:
                        queue.append((member.title, depth + 1))

                    time.sleep(0.25)

                new_pages = pages_per_category.difference(discovered_pages)
                discovered_pages.update(new_pages)
                for page in new_pages:
                    metadata[page] = metadata_per_category[page]

                progress_bar.n = depth
                progress_bar.set_postfix(
                    total_pages=len(discovered_pages),
                    categories=len(visited_categories),
                    queue=len(queue),
                    current_depth=depth
                )

                time.sleep(1)
                break

            except Exception as e:
                outer_retry += 1
                print(
                    f"Outer Retry {outer_retry}/{MAX_RETRIES}\n {category_title} - {e}")
                time.sleep(2 * outer_retry)

    progress_bar.n = depth
    progress_bar.close()
    return discovered_pages, metadata


def extract_entities(data):

    all_entities = {}

    for domain, info in data.items():
        print(f"\nCrawling domain: {domain}")

        discovered_titles, metadata = crawl_categories(info, max_depth=3)

        print(
            f"Discovered {len(discovered_titles)} candidate pages for {domain}")

        all_entities[domain] = {
            "entities": list(discovered_titles),
            "metadata": metadata
        }

    return all_entities


def main():
    ACTORS_ONLY_CONFIG = {
        "Actors": {
            "categories":[], # Categories to crawl
            "must_contain": [], # List of acceptable anchors
            
            "ban_list":[] # List of unacceptable anchors
        }
    }

    actors_data = {}

    for key, values in ACTORS_ONLY_CONFIG.items():
        data = extract_entities({key: values})
        actors_data["entities"] = data[key]["entities"]
        actors_data["metadata"] = data[key]["metadata"]

    with open("actors_entities_discovery.json", 'w') as fp:
        json.dump(actors_data, fp, indent=4)


if __name__ == "__main__":
    main()
