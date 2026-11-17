"""
weather.py
----------
STAGE: Cleaned accidents -> Weather-enriched accidents

WHAT THIS DOES:
For every cleaned accident, looks up the historical weather (daily
average temperature, total precipitation, max wind speed) for that
accident's DATE and approximate LOCATION, using the free Open-Meteo
Historical Weather API. No API key is required.

WHY WE DON'T CALL THE API ONCE PER ACCIDENT ROW:
Calling the API once per accident (potentially hundreds/thousands of
times) would be slow and unnecessary - many accidents happened on the
same day in roughly the same area. Instead we:
  1. Round each accident's latitude/longitude to 1 decimal place
     (~11km precision - a practical "representative location").
  2. Group accidents by (rounded_lat, rounded_lon, date).
  3. Call the API ONCE per unique group.
  4. Join the result back onto every accident in that group.
This is the "simpler practical approach" suggested in the assignment
brief, and it keeps runtime and API usage small without ever inventing
fake weather values - every number returned comes from the real API.

CACHING:
Results are cached to data/staging/weather_cache.csv, so re-running this
script doesn't repeat API calls you've already made.
"""

import logging
import os
import time

import pandas as pd
import requests

from config import CLEANED_ACCIDENTS_FILE, ACCIDENTS_WITH_WEATHER_FILE, WEATHER_CACHE_FILE, LOGS_DIR

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "weather.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_cache():
    if os.path.exists(WEATHER_CACHE_FILE):
        return pd.read_csv(WEATHER_CACHE_FILE, dtype={"rounded_lat": float, "rounded_lon": float})
    return pd.DataFrame(columns=["rounded_lat", "rounded_lon", "accident_date",
                                  "temperature_c", "precipitation_mm", "windspeed_max_kmh"])


def fetch_weather_for_group(lat, lon, date_str):
    """Call Open-Meteo for one (lat, lon, date) combination. Returns a dict or None on failure."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
        "timezone": "Europe/London",
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})
        return {
            "temperature_c": daily.get("temperature_2m_mean", [None])[0],
            "precipitation_mm": daily.get("precipitation_sum", [None])[0],
            "windspeed_max_kmh": daily.get("windspeed_10m_max", [None])[0],
        }
    except requests.exceptions.RequestException as error:
        logging.warning(f"Weather API call failed for ({lat}, {lon}, {date_str}): {error}")
        return None


def main():
    df = pd.read_csv(CLEANED_ACCIDENTS_FILE)

    # Step 1: round coordinates to create a "representative location" per group
    df["rounded_lat"] = df["latitude"].round(1)
    df["rounded_lon"] = df["longitude"].round(1)

    unique_groups = df[["rounded_lat", "rounded_lon", "accident_date"]].drop_duplicates()
    print(f"{len(df)} cleaned accidents -> {len(unique_groups)} unique location/date groups to look up")

    cache_df = load_cache()
    new_results = []
    skipped_cached = 0
    failed_calls = 0

    for _, group in unique_groups.iterrows():
        lat, lon, date_str = group["rounded_lat"], group["rounded_lon"], group["accident_date"]

        already_cached = cache_df[
            (cache_df["rounded_lat"] == lat)
            & (cache_df["rounded_lon"] == lon)
            & (cache_df["accident_date"] == date_str)
        ]
        if not already_cached.empty:
            skipped_cached += 1
            continue  # already have this one from a previous run

        result = fetch_weather_for_group(lat, lon, date_str)
        if result is not None:
            new_results.append({
                "rounded_lat": lat, "rounded_lon": lon, "accident_date": date_str, **result
            })
        else:
            failed_calls += 1
        time.sleep(0.2)  # be polite to the free API - avoid hammering it

    if new_results:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_results)], ignore_index=True)
        cache_df.to_csv(WEATHER_CACHE_FILE, index=False)

    print(f"New weather records fetched: {len(new_results)}")
    print(f"Groups already cached (skipped): {skipped_cached}")
    if failed_calls:
        print(f"WARNING: {failed_calls} API calls failed (check internet connection / see logs/weather.log)")

    # Step 2: join weather back onto every accident row via the group key
    merged = df.merge(cache_df, on=["rounded_lat", "rounded_lon", "accident_date"], how="left")
    merged = merged.drop(columns=["rounded_lat", "rounded_lon"])
    merged.to_csv(ACCIDENTS_WITH_WEATHER_FILE, index=False)

    missing_weather = merged["temperature_c"].isna().sum()
    summary = (
        f"Weather enrichment complete. {len(merged)} rows written to {ACCIDENTS_WITH_WEATHER_FILE}\n"
        f"  Rows missing weather (API failures, if any): {missing_weather}"
    )
    print(summary)
    logging.info(summary)


if __name__ == "__main__":
    main()
