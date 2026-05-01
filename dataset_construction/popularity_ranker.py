"""
Extracts wiki analytics for all given entities and calculates the weighted sum.

Stores the entities list in descending order of score
"""

import pandas as pd
import tqdm
from wiki_analytics import get_page_features
import numpy as np
import json

def percentile(series):
    return series.rank(pct=True)

def categorize_tier(df):
    df["norm_page_views"] = df.groupby("domain")["page_views"].transform(percentile)
    df["norm_content_length"] = df.groupby("domain")["content_length"].transform(percentile)
    df["norm_references_count"] = df.groupby("domain")["references_count"].transform(percentile)

    df["popularity_score"] = 0.5 * df["norm_page_views"] + 0.2 * df["norm_content_length"] + 0.3 * df["norm_references_count"]

    return df
  

def main():
    with open("actors_entities_discovery.json", 'r') as fp:
        data = json.load(fp)

    clusters = {
        "Actors": data["entities"]
    }


    dataset = []

    for domain, entities in clusters.items():
        print(f"Processing {domain}")
        for entity in tqdm.tqdm(entities):
            features = get_page_features(entity, domain)
            if features:
                dataset.append(features)

    df = pd.DataFrame(dataset)
    df = categorize_tier(df)
    df.sort_values(by=["popularity_score"], ascending=[False], ignore_index=True, inplace=True)
    df.to_csv("actors_entities_features_only.csv", index=False)

    with open("actors_entities_only.json", 'w') as fp:
        json.dump(df["entity_name"].tolist(), fp)


if __name__ == "__main__":
    main()
