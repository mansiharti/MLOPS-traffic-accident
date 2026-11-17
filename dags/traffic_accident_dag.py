import os
import sys
from datetime import datetime, timedelta

AIRFLOW_AVAILABLE = False
if sys.platform != "win32":
    try:
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        AIRFLOW_AVAILABLE = True
    except ImportError:
        AIRFLOW_AVAILABLE = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def task_clean_accidents(**context):
    from scripts.pipeline import run_pipeline
    print("Executing Task: Clean and Validate Accidents...")
    clean_df = run_pipeline()
    return f"Successfully cleaned {len(clean_df)} records."


def task_ingest_raw_data(**context):
    from scripts.pipeline import source_path
    raw_csv = source_path()
    print(f"Ingested raw accident data from {raw_csv}")
    return raw_csv


def task_enrich_weather(**context):
    from scripts.weather_enrichment import enrich_weather
    print("Executing Task: Historical Weather Enrichment...")
    enriched_df = enrich_weather()
    return f"Successfully enriched {len(enriched_df)} records with weather telemetry."


def task_load_database(**context):
    from scripts.load_db import load_dataset
    print("Executing Task: Star Schema Warehouse Loading...")
    load_dataset()
    return "Database loaded successfully."


def task_validate_mart(**context):
    mart_count = 0
    fact_count = 0
    checked = False
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "traffic_accidents"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "traffic_password"),
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM accident_analytics;")
            mart_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM fact_accident;")
            fact_count = cur.fetchone()[0]
        conn.close()
        checked = True
    except Exception:
        pass

    if not checked:
        db_path = os.path.join(PROJECT_ROOT, "data", "traffic_accidents.db")
        if os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            mart_count = cur.execute("SELECT COUNT(*) FROM accident_analytics;").fetchone()[0]
            fact_count = cur.execute("SELECT COUNT(*) FROM fact_accident;").fetchone()[0]
            conn.close()
            checked = True

    if mart_count == 0 or fact_count == 0:
        raise ValueError("Data mart or fact table is empty!")

    print(f"Validation Passed: fact_accident has {fact_count:,} rows, accident_analytics has {mart_count:,} rows.")
    return True


default_args = {
    "owner": "data_engineering",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="traffic_accident_severity_pipeline",
        default_args=default_args,
        description="End-to-End Traffic Accident ETL, Weather Enrichment & Mart Loading",
        schedule="@daily",
        catchup=False,
        max_active_runs=1,
        tags=["traffic", "weather", "etl", "star-schema"],
    ) as dag:

        ingest_task = PythonOperator(
            task_id="ingest_raw_data",
            python_callable=task_ingest_raw_data,
        )

        clean_task = PythonOperator(
            task_id="clean_data",
            python_callable=task_clean_accidents,
        )

        enrich_task = PythonOperator(
            task_id="weather_enrichment",
            python_callable=task_enrich_weather,
        )

        load_task = PythonOperator(
            task_id="load_postgresql",
            python_callable=task_load_database,
        )

        validate_task = PythonOperator(
            task_id="validate_mart",
            python_callable=task_validate_mart,
        )

        ingest_task >> clean_task >> enrich_task >> load_task >> validate_task


if __name__ == "__main__":
    print("\n--- Running Airflow Pipeline Tasks Sequentially ---")
    task_ingest_raw_data()
    task_clean_accidents()
    task_enrich_weather()
    task_load_database()
    task_validate_mart()
    print("\n--- All Airflow Tasks Completed Successfully! ---")
