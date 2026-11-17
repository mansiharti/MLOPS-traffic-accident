"""
ingest.py
---------
STAGE: Raw -> Staging

WHAT THIS DOES:
Reads the raw accident CSV exactly as downloaded (no cleaning yet), does
some very light standardisation (column names, basic types), and writes
it out as the "staging" version of the data.

WHY WE SEPARATE RAW FROM STAGING:
- "Raw" should always stay untouched, exactly as it arrived. If something
  goes wrong later, we can always go back to raw and start again.
- "Staging" is where we start shaping the data (consistent column names,
  correct data types) but WITHOUT yet deciding which rows are good or bad.
  That decision belongs to the cleaning step (clean.py), so it's kept
  separate and easy to explain on its own.

This script does NOT reject or fix bad values - it only standardises
structure. That keeps each layer responsible for exactly one job.
"""

import logging
import os

import pandas as pd

from config import RAW_ACCIDENTS_FILE, STAGING_ACCIDENTS_FILE, LOGS_DIR

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    print(f"Reading raw file: {RAW_ACCIDENTS_FILE}")
    df = pd.read_csv(RAW_ACCIDENTS_FILE)

    # Standardise column names: lowercase, no leading/trailing spaces.
    # (Real STATS19 downloads sometimes have inconsistent capitalisation.)
    df.columns = [c.strip().lower() for c in df.columns]

    # Standardise text-like columns: strip stray whitespace.
    text_columns = [c for c in df.columns if df[c].dtype == object]
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
        # str(NaN) turns into the string "nan" - convert that back to a
        # real empty value so later steps can detect missing data correctly.
        df[col] = df[col].replace({"nan": "", "None": ""})

    df.to_csv(STAGING_ACCIDENTS_FILE, index=False)

    msg = f"Ingest complete. {len(df)} rows written to {STAGING_ACCIDENTS_FILE}"
    print(msg)
    logging.info(msg)


if __name__ == "__main__":
    main()
