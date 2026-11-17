"""
load_database.py
-----------------
STAGE: Cleaned/weather-enriched data -> PostgreSQL

WHAT THIS DOES:
1. Connects to PostgreSQL and makes sure the tables exist (runs schema.sql).
2. Loads the weather-enriched, cleaned accidents into the `accidents` table.
3. Loads the rejected records into the `rejected_accidents` table (so
   invalid data is stored separately, not thrown away - useful for
   auditing what got rejected and why).
4. Loads the unique weather lookups into the `weather` table.
5. Builds the `accident_analytics` data mart: a small pre-aggregated
   summary table (accidents grouped by date, road type, and severity)
   that the dashboard can query quickly instead of aggregating raw rows
   every time.

WHY WE USE "IF EXISTS -> DELETE THEN INSERT" INSTEAD OF constantly
APPENDING: this script is designed to be safely re-runnable. Every time
you run the full pipeline, it replaces the tables' contents with the
latest cleaned data rather than duplicating rows.
"""

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

from config import (
    ACCIDENTS_WITH_WEATHER_FILE,
    REJECTED_ACCIDENTS_FILE,
    WEATHER_CACHE_FILE,
    DB_CONNECTION_STRING,
    LOGS_DIR,
    PROJECT_ROOT,
)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "load_database.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

SCHEMA_FILE = os.path.join(PROJECT_ROOT, "sql", "schema.sql")


def run_schema(engine):
    """Create all tables from sql/schema.sql if they don't already exist."""
    with open(SCHEMA_FILE, "r") as f:
        schema_sql = f.read()

    with engine.begin() as connection:
        # Run each CREATE statement separately (splitting on ';' is fine here
        # because schema.sql only contains simple statements, no semicolons
        # inside strings/functions).
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(text(statement))
    print("Schema check complete (tables created if they didn't already exist).")


def load_accidents(engine):
    df = pd.read_csv(ACCIDENTS_WITH_WEATHER_FILE)

    # Keep only the columns that exist in the accidents table, in case the
    # cleaned CSV has extra helper columns.
    accidents_columns = [
        "accident_index", "accident_date", "accident_time", "day_of_week",
        "accident_severity", "number_of_vehicles", "number_of_casualties",
        "road_type", "speed_limit", "light_conditions", "road_surface_conditions",
        "latitude", "longitude", "local_authority",
        "temperature_c", "precipitation_mm", "windspeed_max_kmh",
    ]
    df = df[[c for c in accidents_columns if c in df.columns]]

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM accidents"))
    df.to_sql("accidents", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into accidents table.")


def load_rejected(engine):
    df = pd.read_csv(REJECTED_ACCIDENTS_FILE)
    rejected_columns = [
        "accident_index", "accident_date", "accident_time", "accident_severity",
        "road_type", "latitude", "longitude", "rejection_reasons",
    ]
    df = df[[c for c in rejected_columns if c in df.columns]]

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM rejected_accidents"))
    df.to_sql("rejected_accidents", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into rejected_accidents table.")


def load_weather(engine):
    if not os.path.exists(WEATHER_CACHE_FILE):
        print("No weather cache file found - skipping weather table load.")
        return
    df = pd.read_csv(WEATHER_CACHE_FILE)
    df = df.rename(columns={"accident_date": "weather_date"})

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM weather"))
    df.to_sql("weather", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into weather table.")


def build_analytics_mart(engine):
    """
    Build the analytical data mart directly in SQL: group the accidents
    table by date, road type, and severity, and store the aggregated
    counts in accident_analytics. This is the "analytical data mart"
    required by the assignment.
    """
    mart_sql = """
        INSERT INTO accident_analytics (
            accident_date, road_type, accident_severity,
            total_accidents, total_vehicles_involved, total_casualties,
            avg_temperature_c, avg_precipitation_mm
        )
        SELECT
            accident_date,
            road_type,
            accident_severity,
            COUNT(*) AS total_accidents,
            SUM(number_of_vehicles) AS total_vehicles_involved,
            SUM(number_of_casualties) AS total_casualties,
            AVG(temperature_c) AS avg_temperature_c,
            AVG(precipitation_mm) AS avg_precipitation_mm
        FROM accidents
        GROUP BY accident_date, road_type, accident_severity;
    """
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM accident_analytics"))
        connection.execute(text(mart_sql))
    print("Analytical data mart (accident_analytics) rebuilt.")


def main():
    engine = create_engine(DB_CONNECTION_STRING)

    run_schema(engine)
    load_accidents(engine)
    load_rejected(engine)
    load_weather(engine)
    build_analytics_mart(engine)

    logging.info("Database load complete.")
    print("\nAll done! Your PostgreSQL database now has: accidents, "
          "rejected_accidents, weather, and accident_analytics tables.")


if __name__ == "__main__":
    main()
