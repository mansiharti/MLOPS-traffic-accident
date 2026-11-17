import json
import os
import pandas as pd


# UK Department for Transport collision weather-condition codes.
RECORDED_WEATHER_MAP = {
    1: ("Fine no high winds", 0.0, 0.0, 0.0),
    2: ("Raining no high winds", 1.0, 1.0, 0.0),
    3: ("Snowing no high winds", 1.0, 0.0, 1.0),
    4: ("Fine with high winds", 0.0, 0.0, 0.0),
    5: ("Raining with high winds", 1.0, 1.0, 0.0),
    6: ("Snowing with high winds", 1.0, 0.0, 1.0),
    7: ("Fog or mist", 0.0, 0.0, 0.0),
    8: ("Other", 0.0, 0.0, 0.0),
    9: ("Unknown", 0.0, 0.0, 0.0),
    -1: ("Unknown", 0.0, 0.0, 0.0),
}


def enrich_weather(input_csv="data/output/accidents_cleaned.csv", output_csv="data/output/accidents_enriched.csv"):
    """Use weather observed at each UK collision; the supplied Berlin CSV is not used."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found at {input_csv}. Run pipeline.py first.")
    df = pd.read_csv(input_csv)
    if "recorded_weather_code" not in df:
        raise ValueError("The UK collision weather field is missing from the cleaned dataset.")
    codes = pd.to_numeric(df["recorded_weather_code"], errors="coerce").fillna(9).astype(int)
    mapped = codes.map(lambda c: RECORDED_WEATHER_MAP.get(c, RECORDED_WEATHER_MAP[9]))
    df["weather_code"] = codes
    df["weather_condition"] = mapped.map(lambda item: item[0])
    df["precipitation"] = mapped.map(lambda item: item[1])
    df["rain"] = mapped.map(lambda item: item[2])
    df["snowfall"] = mapped.map(lambda item: item[3])
    df["wind_speed"] = codes.isin([4, 5, 6]).map({True: 25.0, False: 10.0})
    # Temperature was not supplied in the official collision extract.
    df["temperature"] = 0.0
    df.to_csv(output_csv, index=False)
    cache_path = os.path.join("data", "output", "weather_cache.json")
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump({"source": "UK Department for Transport collision weather_conditions field", "year": 2025}, cache_file, indent=2)
    print(f"[Weather Enrichment] Used recorded collision-scene weather for {len(df):,} UK 2025 records.")
    print(f"[Weather Enrichment] Output saved to: {output_csv}")
    print(df["weather_condition"].value_counts().to_string())
    return df


if __name__ == "__main__":
    enrich_weather()
