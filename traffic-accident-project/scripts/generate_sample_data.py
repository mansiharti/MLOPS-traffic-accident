"""
generate_sample_data.py
------------------------
Creates a SAMPLE raw accident CSV file that has the SAME columns and
general shape as the real UK STATS19 "Road Safety" collision data.

WHY THIS SCRIPT EXISTS:
The real dataset is downloaded from gov.uk:
    https://www.gov.uk/government/statistics/road-safety-data
This build environment cannot reach gov.uk, so this script generates a
realistic stand-in dataset so the ENTIRE pipeline can be built, run, and
demonstrated end-to-end right now.

WHAT YOU SHOULD DO:
Run this script once to get a working demo dataset. Later, replace
data/raw/accidents_raw.csv with the real STATS19 collision CSV you
download yourself (keeping the same column names, or adjusting
clean.py's COLUMN_MAP if the real file's column names differ slightly).
The rest of the pipeline does not need to change.

This script also deliberately injects some BAD rows (missing IDs, bad
severity, missing coordinates, duplicates) so that clean.py has
something real to catch and reject - this proves your data-quality
checks actually work.
"""

import random
from datetime import datetime, timedelta

import pandas as pd

from config import RAW_ACCIDENTS_FILE

random.seed(42)  # fixed seed = same "random" data every time you run this

# Real STATS19 category values (kept authentic even though the rows are synthetic)
ROAD_TYPES = ["Single carriageway", "Dual carriageway", "Roundabout",
              "One way street", "Slip road"]
SPEED_LIMITS = [20, 30, 40, 50, 60, 70]
LIGHT_CONDITIONS = ["Daylight", "Darkness - lights lit",
                     "Darkness - no lighting", "Darkness - lights unlit"]
ROAD_SURFACE = ["Dry", "Wet or damp", "Snow", "Frost or ice"]
SEVERITIES = ["Fatal", "Serious", "Slight"]
# Slight accidents are far more common than fatal - weight accordingly
SEVERITY_WEIGHTS = [0.03, 0.22, 0.75]
LOCAL_AUTHORITIES = ["Leeds", "Manchester", "Birmingham", "Bristol",
                      "Sheffield", "Liverpool", "Nottingham", "Newcastle"]

N_GOOD_ROWS = 650
N_BAD_ROWS = 60  # deliberately invalid rows for data-quality testing

START_DATE = datetime(2024, 1, 1)


def random_date_in_2024():
    days_offset = random.randint(0, 365)
    return START_DATE + timedelta(days=days_offset)


def random_uk_coordinates():
    # Rough Great Britain bounding box
    lat = round(random.uniform(50.5, 55.5), 6)
    lon = round(random.uniform(-4.5, 0.5), 6)
    return lat, lon


def build_good_row(i):
    date = random_date_in_2024()
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    lat, lon = random_uk_coordinates()
    return {
        "accident_index": f"2024{i:07d}",
        "accident_date": date.strftime("%Y-%m-%d"),
        "accident_time": f"{hour:02d}:{minute:02d}",
        "day_of_week": date.strftime("%A"),
        "accident_severity": random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0],
        "number_of_vehicles": random.randint(1, 4),
        "number_of_casualties": random.randint(1, 3),
        "road_type": random.choice(ROAD_TYPES),
        "speed_limit": random.choice(SPEED_LIMITS),
        "light_conditions": random.choice(LIGHT_CONDITIONS),
        "road_surface_conditions": random.choice(ROAD_SURFACE),
        "latitude": lat,
        "longitude": lon,
        "local_authority": random.choice(LOCAL_AUTHORITIES),
    }


def build_bad_rows():
    """Deliberately broken rows, one issue type at a time, for QC testing."""
    bad_rows = []

    # 1) Missing accident_index
    row = build_good_row(9000001)
    row["accident_index"] = ""
    bad_rows.append(row)

    # 2) Missing date
    row = build_good_row(9000002)
    row["accident_date"] = ""
    bad_rows.append(row)

    # 3) Invalid severity
    row = build_good_row(9000003)
    row["accident_severity"] = "Unknown"
    bad_rows.append(row)

    # 4) Invalid coordinates (0,0 - "null island", a classic bad-GPS value)
    row = build_good_row(9000004)
    row["latitude"] = 0
    row["longitude"] = 0
    bad_rows.append(row)

    # 5) Out-of-range coordinates (outside the UK entirely)
    row = build_good_row(9000005)
    row["latitude"] = 200.0
    row["longitude"] = 999.0
    bad_rows.append(row)

    # 6) Missing important field (road_type)
    row = build_good_row(9000006)
    row["road_type"] = ""
    bad_rows.append(row)

    # 7) Duplicate accident_index (copy an existing good ID twice)
    dup_row_a = build_good_row(9000007)
    dup_row_a["accident_index"] = "2024_DUPLICATE_ID"
    dup_row_b = build_good_row(9000008)
    dup_row_b["accident_index"] = "2024_DUPLICATE_ID"
    bad_rows.append(dup_row_a)
    bad_rows.append(dup_row_b)

    # Fill the remaining bad-row quota by repeating a mix of the issues above
    issue_builders = [
        lambda r: {**r, "accident_index": ""},
        lambda r: {**r, "accident_date": ""},
        lambda r: {**r, "accident_severity": "Unclassified"},
        lambda r: {**r, "latitude": None, "longitude": None},
        lambda r: {**r, "road_type": ""},
    ]
    next_id = 9000009
    while len(bad_rows) < N_BAD_ROWS:
        base = build_good_row(next_id)
        builder = random.choice(issue_builders)
        bad_rows.append(builder(base))
        next_id += 1

    return bad_rows


def main():
    good_rows = [build_good_row(i) for i in range(1, N_GOOD_ROWS + 1)]
    bad_rows = build_bad_rows()

    all_rows = good_rows + bad_rows
    random.shuffle(all_rows)  # mix good and bad rows like a real messy dataset

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_ACCIDENTS_FILE, index=False)

    print(f"Sample raw dataset created: {RAW_ACCIDENTS_FILE}")
    print(f"  Total rows: {len(df)}")
    print(f"  Good rows (approx): {N_GOOD_ROWS}")
    print(f"  Deliberately bad rows (approx): {N_BAD_ROWS}")


if __name__ == "__main__":
    main()
