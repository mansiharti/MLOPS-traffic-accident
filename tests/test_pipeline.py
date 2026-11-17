import os
import sqlite3
import unittest
import pandas as pd

from scripts.pipeline import (
    SEVERITY_MAP,
    UK_LAT_MAX,
    UK_LAT_MIN,
    UK_LON_MAX,
    UK_LON_MIN,
    VALID_SEVERITIES,
    code_label,
)


class TestTrafficAccidentDataPipeline(unittest.TestCase):

    def test_01_severity_mapping(self):
        self.assertEqual(SEVERITY_MAP["1"], "Fatal")
        self.assertEqual(SEVERITY_MAP["2"], "Serious")
        self.assertEqual(SEVERITY_MAP["3"], "Slight")
        self.assertEqual(VALID_SEVERITIES, {"Fatal", "Serious", "Slight"})

    def test_02_uk_geographic_bounds(self):
        self.assertEqual(UK_LAT_MIN, 49.0)
        self.assertEqual(UK_LAT_MAX, 61.5)
        self.assertEqual(UK_LON_MIN, -8.5)
        self.assertEqual(UK_LON_MAX, 2.5)

    def test_03_code_label_lookup(self):
        mapping = {"1": "Daylight", "4": "Darkness"}
        self.assertEqual(code_label("1", mapping), "Daylight")
        self.assertEqual(code_label("99", mapping, default="Unknown"), "Unknown")

    def test_04_cleaned_and_enriched_files_exist(self):
        cleaned_path = os.path.join("data", "output", "accidents_cleaned.csv")
        enriched_path = os.path.join("data", "output", "accidents_enriched.csv")
        rejected_path = os.path.join("data", "output", "rejected_records.csv")

        self.assertTrue(os.path.exists(cleaned_path), "accidents_cleaned.csv should exist")
        self.assertTrue(os.path.exists(enriched_path), "accidents_enriched.csv should exist")
        self.assertTrue(os.path.exists(rejected_path), "rejected_records.csv should exist")

    def test_05_warehouse_database_integrity(self):
        db_path = os.path.join("data", "traffic_accidents.db")
        self.assertTrue(os.path.exists(db_path), "SQLite warehouse database should exist")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        expected_tables = [
            "dim_date",
            "dim_location",
            "dim_road",
            "dim_weather",
            "dim_severity",
            "fact_accident",
            "accident_analytics",
        ]
        for tbl in expected_tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, f"Table {tbl} should not be empty")

        cur.execute("SELECT COUNT(*) FROM fact_accident;")
        fact_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM accident_analytics;")
        mart_count = cur.fetchone()[0]
        self.assertEqual(fact_count, mart_count, "Fact and mart row counts must match")

        conn.close()


if __name__ == "__main__":
    unittest.main()

