import os
import sys
import time
import sqlite3
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline import run_pipeline
from scripts.weather_enrichment import enrich_weather
from scripts.load_db import load_dataset


def run_full_pipeline():
    start_time = time.time()
    print("=" * 65)
    print("[*] TRAFFIC ACCIDENT SEVERITY & WEATHER ANALYTICS PIPELINE")
    print("=" * 65)

    print("\n[STEP 1/3] Ingesting & Cleaning Raw Accidents...")
    clean_df = run_pipeline()

    print("\n[STEP 2/3] Enriching with Historical Weather from Open-Meteo...")
    enriched_df = enrich_weather()

    print("\n[STEP 3/3] Loading Dimensional Star Schema & Analytical Mart...")
    load_dataset()

    db_path = os.path.join("data", "traffic_accidents.db")
    print("\n" + "=" * 65)
    print("[+] PIPELINE EXECUTION SUMMARY & HEALTH REPORT")
    print("=" * 65)

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        total_facts = cur.execute("SELECT COUNT(*) FROM fact_accident;").fetchone()[0]
        total_mart = cur.execute("SELECT COUNT(*) FROM accident_analytics;").fetchone()[0]
        fatal_count = cur.execute("SELECT COUNT(*) FROM accident_analytics WHERE accident_severity = 'Fatal';").fetchone()[0]
        serious_count = cur.execute("SELECT COUNT(*) FROM accident_analytics WHERE accident_severity = 'Serious';").fetchone()[0]
        slight_count = cur.execute("SELECT COUNT(*) FROM accident_analytics WHERE accident_severity = 'Slight';").fetchone()[0]
        total_casualties = cur.execute("SELECT SUM(number_of_casualties) FROM accident_analytics;").fetchone()[0]
        conn.close()

        print(f"Total Collisions Processed : {total_mart:,}")
        print(f"Total Casualties Recorded  : {total_casualties:,}")
        print(f"Severity Distribution      : Fatal: {fatal_count} | Serious: {serious_count} | Slight: {slight_count}")
        print(f"Analytical Database        : {db_path}")

    elapsed = time.time() - start_time
    print(f"Execution Completed in     : {elapsed:.2f} seconds")
    print("=" * 65)
    print("\n[>>] TO LAUNCH THE INTERACTIVE STREAMLIT DASHBOARD:")
    print("     streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_full_pipeline()
