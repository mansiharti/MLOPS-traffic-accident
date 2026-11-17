import os
import pandas as pd

VALID_SEVERITIES = {"Fatal", "Serious", "Slight"}
UK_LAT_MIN, UK_LAT_MAX = 49.0, 61.5
UK_LON_MIN, UK_LON_MAX = -8.5, 2.5
SEVERITY_MAP = {"1": "Fatal", "2": "Serious", "3": "Slight"}
ROAD_TYPE_MAP = {"1": "Roundabout", "2": "One way street", "3": "Dual carriageway", "6": "Single carriageway", "7": "Slip road", "9": "Unknown"}
LIGHT_MAP = {"1": "Daylight", "4": "Darkness - lights lit", "5": "Darkness - lights unlit", "6": "Darkness - no lighting", "7": "Darkness - lighting unknown"}
SURFACE_MAP = {"1": "Dry", "2": "Wet or damp", "3": "Snow", "4": "Frost or ice", "5": "Flood over 3cm deep", "6": "Oil or diesel", "7": "Mud", "9": "Unknown"}


def source_path():
    path = os.path.join("data", "raw", "uk_2025", "collisions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"UK collision dataset not found: {path}")
    return path


def detail_source_path(filename):
    path = os.path.join("data", "raw", "uk_2025", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"UK detail dataset not found: {path}")
    return path


def detail_counts(filename, count_column):
    source = pd.read_csv(detail_source_path(filename), dtype=str, keep_default_na=False)
    if "collision_index" not in source.columns:
        raise ValueError(f"{filename} is missing required column: collision_index")
    counts = source.loc[source["collision_index"].str.strip().ne("")].groupby("collision_index").size()
    return counts.rename(count_column)


def code_label(value, mapping, default="Unknown"):
    return mapping.get(str(value).strip(), default)


def normalise_collisions(source):
    """Convert the official DfT 2025 collision layout to the warehouse layout."""
    required = {"collision_index", "date", "time", "collision_severity", "number_of_vehicles", "number_of_casualties", "road_type", "speed_limit", "light_conditions", "road_surface_conditions", "latitude", "longitude", "local_authority_district", "weather_conditions"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Collision source is missing required columns: {sorted(missing)}")
    authority_column = "local_authority_ons_district" if "local_authority_ons_district" in source.columns else "local_authority_district"
    return pd.DataFrame({
        "accident_index": source["collision_index"].astype(str).str.strip(),
        "accident_date": pd.to_datetime(source["date"], dayfirst=True, errors="coerce"),
        "accident_time": source["time"].astype(str).str.strip(),
        "accident_severity": source["collision_severity"].map(SEVERITY_MAP),
        "number_of_vehicles": pd.to_numeric(source["number_of_vehicles"], errors="coerce"),
        "number_of_casualties": pd.to_numeric(source["number_of_casualties"], errors="coerce"),
        "road_type": source["road_type"].map(lambda v: code_label(v, ROAD_TYPE_MAP)),
        "speed_limit": pd.to_numeric(source["speed_limit"], errors="coerce"),
        "light_conditions": source["light_conditions"].map(lambda v: code_label(v, LIGHT_MAP)),
        "road_surface_conditions": source["road_surface_conditions"].map(lambda v: code_label(v, SURFACE_MAP)),
        "latitude": pd.to_numeric(source["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(source["longitude"], errors="coerce"),
        "local_authority": source[authority_column].astype(str).str.strip(),
        "recorded_weather_code": pd.to_numeric(source["weather_conditions"], errors="coerce").fillna(9).astype(int),
    })


def run_pipeline():
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = source_path()
    print(f"-> Found UK 2025 collision dataset: {csv_path}")
    df = normalise_collisions(pd.read_csv(csv_path, dtype=str, keep_default_na=False))
    vehicle_counts = detail_counts("vehicles.csv", "vehicle_detail_count")
    casualty_counts = detail_counts("casualties.csv", "casualty_detail_count")
    df = df.merge(vehicle_counts, left_on="accident_index", right_index=True, how="left")
    df = df.merge(casualty_counts, left_on="accident_index", right_index=True, how="left")
    df["number_of_vehicles"] = df["vehicle_detail_count"].fillna(df["number_of_vehicles"])
    df["number_of_casualties"] = df["casualty_detail_count"].fillna(df["number_of_casualties"])
    df = df.drop(columns=["vehicle_detail_count", "casualty_detail_count"])
    print("-> Joined vehicles.csv and casualties.csv to collision records")
    total_raw_rows = len(df)
    duplicate_mask = df.duplicated(subset=["accident_index"], keep="first") & df["accident_index"].ne("")
    valid_time = df["accident_time"].str.match(r"^([01]\d|2[0-3]):[0-5]\d$", na=False)
    valid_coordinates = df["latitude"].between(UK_LAT_MIN, UK_LAT_MAX) & df["longitude"].between(UK_LON_MIN, UK_LON_MAX) & ~(df["latitude"].eq(0) & df["longitude"].eq(0))
    valid_mask = ~duplicate_mask & df["accident_index"].ne("") & df["accident_date"].notna() & valid_time & df["accident_severity"].isin(VALID_SEVERITIES) & valid_coordinates & df["number_of_vehicles"].notna() & df["number_of_casualties"].notna()
    rejected_df = df.loc[~valid_mask].copy()
    rejected_df["rejection_reason"] = "failed required identifier, date/time, severity, count, or UK-coordinate validation"
    clean_df = df.loc[valid_mask].copy()
    clean_df["accident_date"] = clean_df["accident_date"].dt.strftime("%Y-%m-%d")
    clean_df["accident_hour"] = clean_df["accident_time"].str[:2].astype(int)
    clean_df["day_of_week"] = pd.to_datetime(clean_df["accident_date"]).dt.day_name()
    for column in ["number_of_vehicles", "number_of_casualties"]:
        clean_df[column] = clean_df[column].astype(int)
    clean_df["speed_limit"] = clean_df["speed_limit"].fillna(30).astype(int)
    clean_df["latitude"] = clean_df["latitude"].round(6)
    clean_df["longitude"] = clean_df["longitude"].round(6)
    cleaned_path = os.path.join(output_dir, "accidents_cleaned.csv")
    rejected_path = os.path.join(output_dir, "rejected_records.csv")
    clean_df.to_csv(cleaned_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)
    print("=" * 50)
    print("TRAFFIC ACCIDENT PIPELINE: UK 2025 CLEANING SUMMARY")
    print("=" * 50)
    print(f"Rows before cleaning: {total_raw_rows:,}")
    print(f"Rows after cleaning:  {len(clean_df):,}")
    print(f"Duplicates removed:   {int(duplicate_mask.sum()):,}")
    print(f"Invalid records:      {len(rejected_df):,}")
    print(f"Cleaned output: {cleaned_path}")
    return clean_df


if __name__ == "__main__":
    run_pipeline()
