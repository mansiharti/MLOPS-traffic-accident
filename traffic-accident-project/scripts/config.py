"""
config.py
----------
One simple place for all settings used across the pipeline scripts.
Keeping settings here (instead of hard-coding them in every file) means
you only need to change a value in ONE place if something changes
(e.g. your Postgres password).

Beginner note: in a real company project these values would usually come
from environment variables or a .env file so secrets are never stored in
code. For a student project, plain variables here are simpler to read and
explain, so that's what we use.
"""

import os

# ---------------------------------------------------------------------
# Folder paths (all relative to the project root)
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
REJECTED_DIR = os.path.join(PROJECT_ROOT, "data", "rejected")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

RAW_ACCIDENTS_FILE = os.path.join(RAW_DIR, "accidents_raw.csv")
STAGING_ACCIDENTS_FILE = os.path.join(STAGING_DIR, "accidents_staging.csv")
CLEANED_ACCIDENTS_FILE = os.path.join(CLEANED_DIR, "accidents_cleaned.csv")
REJECTED_ACCIDENTS_FILE = os.path.join(REJECTED_DIR, "accidents_rejected.csv")
WEATHER_CACHE_FILE = os.path.join(STAGING_DIR, "weather_cache.csv")
ACCIDENTS_WITH_WEATHER_FILE = os.path.join(CLEANED_DIR, "accidents_with_weather.csv")

# ---------------------------------------------------------------------
# PostgreSQL connection settings
# Change these to match your local PostgreSQL installation.
# ---------------------------------------------------------------------
DB_HOST = os.environ.get("TRAFFIC_DB_HOST", "localhost")
DB_PORT = os.environ.get("TRAFFIC_DB_PORT", "5432")
DB_NAME = os.environ.get("TRAFFIC_DB_NAME", "traffic_accidents")
DB_USER = os.environ.get("TRAFFIC_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("TRAFFIC_DB_PASSWORD", "postgres")

# SQLAlchemy connection string built from the settings above
DB_CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------------------------------------
# Data-quality thresholds (used by clean.py)
# UK bounding box - roughly covers Great Britain including islands
# ---------------------------------------------------------------------
UK_LAT_MIN, UK_LAT_MAX = 49.8, 60.9
UK_LON_MIN, UK_LON_MAX = -8.6, 1.8

VALID_SEVERITIES = {"Fatal", "Serious", "Slight"}
