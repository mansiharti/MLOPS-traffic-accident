import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests


INPUT_CSV = os.path.join("data", "output", "accidents_enriched.csv")
OUTPUT_JSON = os.path.join("data", "authority_names.json")


def resolve_authority(item):
    code, latitude, longitude = item
    try:
        response = requests.get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={"latitude": latitude, "longitude": longitude, "localityLanguage": "en"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        name = payload.get("city") or payload.get("locality") or payload.get("principalSubdivision")
        return code, name or code
    except requests.RequestException:
        return code, code


def main():
    data = pd.read_csv(INPUT_CSV)
    locations = data.dropna(subset=["local_authority", "latitude", "longitude"]).drop_duplicates("local_authority")
    items = locations[["local_authority", "latitude", "longitude"]].itertuples(index=False, name=None)
    with ThreadPoolExecutor(max_workers=8) as executor:
        names = dict(executor.map(resolve_authority, items))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as output_file:
        json.dump(names, output_file, indent=2, sort_keys=True)
    print(f"Wrote {len(names)} authority names to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()