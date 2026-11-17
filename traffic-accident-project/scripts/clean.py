"""
clean.py
--------
STAGE: Staging -> Cleaned / Rejected

WHAT THIS DOES:
Reads the staging accident data and runs a series of simple data-quality
checks on every row. Rows that pass ALL checks go to the "cleaned" layer.
Rows that fail ANY check go to the "rejected" layer, together with a
column explaining WHY they were rejected.

This maps directly to Part 1 requirements:
  - "Performs data-quality checks"
  - "Stores rejected/invalid records separately"

WHY WE CHECK ONE RULE AT A TIME:
Each check below is a small, independent function that returns True/False
for a single row. This keeps the logic easy to read, easy to test, and
easy to explain individually in a viva ("this function checks X, this one
checks Y").
"""

import logging
import os

import pandas as pd

from config import (
    STAGING_ACCIDENTS_FILE,
    CLEANED_ACCIDENTS_FILE,
    REJECTED_ACCIDENTS_FILE,
    LOGS_DIR,
    UK_LAT_MIN,
    UK_LAT_MAX,
    UK_LON_MIN,
    UK_LON_MAX,
    VALID_SEVERITIES,
)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "clean.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

REQUIRED_FIELDS = ["road_type", "number_of_vehicles", "number_of_casualties"]


def is_missing(value):
    """Treat empty string, whitespace, and real NaN as 'missing'."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def check_missing_id(row):
    return not is_missing(row.get("accident_index"))


def check_missing_datetime(row):
    return not is_missing(row.get("accident_date")) and not is_missing(row.get("accident_time"))


def check_valid_severity(row):
    return row.get("accident_severity") in VALID_SEVERITIES


def check_valid_coordinates(row):
    lat, lon = row.get("latitude"), row.get("longitude")
    if is_missing(lat) or is_missing(lon):
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (ValueError, TypeError):
        return False
    if lat == 0 and lon == 0:
        return False  # classic bad-GPS placeholder ("null island")
    return UK_LAT_MIN <= lat <= UK_LAT_MAX and UK_LON_MIN <= lon <= UK_LON_MAX


def check_required_fields(row):
    for field in REQUIRED_FIELDS:
        if is_missing(row.get(field)):
            return False
    return True


# Each rule: (short reason text, check function that returns True if row PASSES)
RULES = [
    ("missing_accident_id", check_missing_id),
    ("missing_date_or_time", check_missing_datetime),
    ("invalid_severity", check_valid_severity),
    ("invalid_coordinates", check_valid_coordinates),
    ("missing_required_field", check_required_fields),
]


def find_rejection_reasons(row):
    """Run every rule against a row; return a list of reasons it failed (empty if it passed everything)."""
    reasons = [reason for reason, check_fn in RULES if not check_fn(row)]
    return reasons


def main():
    # keep_default_na=False + na_values=[] stops pandas turning "" into NaN,
    # so a missing accident_index reads back as "" (not NaN) and our checks
    # below can tell "missing" apart from "actually the text NaN".
    df = pd.read_csv(STAGING_ACCIDENTS_FILE, dtype=str, keep_default_na=False, na_values=[])
    # Convert numeric-looking columns back for the checks that need numbers,
    # while keeping the original text version for anything that failed to convert.
    df["number_of_vehicles_num"] = pd.to_numeric(df["number_of_vehicles"], errors="coerce")
    df["number_of_casualties_num"] = pd.to_numeric(df["number_of_casualties"], errors="coerce")

    reasons_list = []
    for _, row in df.iterrows():
        reasons_list.append(find_rejection_reasons(row))

    df["rejection_reasons"] = [", ".join(r) if r else "" for r in reasons_list]
    df["is_valid"] = [len(r) == 0 for r in reasons_list]

    # --- Duplicate check (needs the whole table, not just one row) ---
    # Only flag TRUE duplicates: same non-empty accident_index appearing more than once.
    non_empty_ids = df["accident_index"].fillna("").astype(str).str.strip() != ""
    duplicate_mask = df.duplicated(subset=["accident_index"], keep="first") & non_empty_ids
    df.loc[duplicate_mask, "rejection_reasons"] = df.loc[duplicate_mask, "rejection_reasons"].apply(
        lambda existing: (existing + ", " if existing else "") + "duplicate_accident_id"
    )
    df.loc[duplicate_mask, "is_valid"] = False

    cleaned_df = df[df["is_valid"]].drop(columns=["rejection_reasons", "is_valid",
                                                     "number_of_vehicles_num", "number_of_casualties_num"])
    rejected_df = df[~df["is_valid"]].drop(columns=["is_valid",
                                                       "number_of_vehicles_num", "number_of_casualties_num"])

    cleaned_df.to_csv(CLEANED_ACCIDENTS_FILE, index=False)
    rejected_df.to_csv(REJECTED_ACCIDENTS_FILE, index=False)

    summary = (
        f"Cleaning complete.\n"
        f"  Total rows read:    {len(df)}\n"
        f"  Cleaned (valid):    {len(cleaned_df)} -> {CLEANED_ACCIDENTS_FILE}\n"
        f"  Rejected (invalid): {len(rejected_df)} -> {REJECTED_ACCIDENTS_FILE}"
    )
    print(summary)
    logging.info(summary)

    if len(rejected_df) > 0:
        print("\nRejection reason breakdown:")
        reason_counts = (
            rejected_df["rejection_reasons"].str.split(", ").explode().value_counts()
        )
        print(reason_counts.to_string())


if __name__ == "__main__":
    main()
